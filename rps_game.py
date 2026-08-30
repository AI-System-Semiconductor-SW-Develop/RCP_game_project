"""Two-player rock-paper-scissors game using a camera and TensorRT.

SPACE starts a three-second round.  Put one hand on each side of the screen.
Q or ESC closes the game.
"""

from collections import Counter, deque
from pathlib import Path
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (creates the CUDA context)

from trt_module import TRTInferenceEngine


ROOT = Path(__file__).resolve().parent
ENGINE_PATH = ROOT / "rps_game_yolo11n_fp16.engine"
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
STABLE_FRAMES = 7

PLAYER_1_WINS = "PLAYER 1 WINS"
PLAYER_2_WINS = "PLAYER 2 WINS"
DRAW = "DRAW"

CLASS_NAMES = {0: "SCISSORS", 1: "ROCK", 2: "PAPER"}
CLASS_COLORS = {
    0: (255, 180, 0),
    1: (0, 80, 255),
    2: (0, 210, 80),
}


def configure_camera(camera):
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def make_histories(count=2):
    return [deque(maxlen=STABLE_FRAMES) for _ in range(count)]


def clear_histories(histories):
    for history in histories:
        history.clear()


def record_detection(history, detection):
    history.append(None if detection is None else detection["class_id"])


def apply_round_score(scores, round_result):
    if round_result == PLAYER_1_WINS:
        scores[0] += 1
    elif round_result == PLAYER_2_WINS:
        scores[1] += 1


def letterbox(image, size):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (resized_w, resized_h))
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2
    result = cv2.copyMakeBorder(
        resized,
        pad_y,
        size - resized_h - pad_y,
        pad_x,
        size - resized_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return result, scale, pad_x, pad_y


def nms(boxes, scores, threshold):
    if not len(boxes):
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = order[0]
        keep.append(index)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[index] + areas[order[1:]] - intersection + 1e-6
        order = order[np.where(intersection / union <= threshold)[0] + 1]
    return keep


class RPSDetector:
    def __init__(self, engine_path):
        self.engine = TRTInferenceEngine(str(engine_path))
        input_shape = tuple(self.engine.tensors[self.engine.input_name]["shape"])
        self.input_dtype = self.engine.tensors[self.engine.input_name]["host"].dtype
        self.image_size = int(input_shape[-1])
        self.output_shape = tuple(self.engine.tensors[self.engine.output_name]["shape"])

    def detect(self, frame):
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, scale, pad_x, pad_y = letterbox(rgb, self.image_size)
        tensor = padded.astype(self.input_dtype) / np.array(255.0, dtype=self.input_dtype)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])

        raw = self.engine.infer(tensor).reshape(self.output_shape)
        raw = raw[0]
        # Ultralytics detection output is normally (4 + classes, anchors).
        predictions = raw.T if raw.shape[0] <= raw.shape[1] else raw
        if predictions.shape[1] < 7:
            raise RuntimeError(f"Unexpected model output shape: {self.output_shape}")

        scores_by_class = predictions[:, 4:7]
        class_ids = scores_by_class.argmax(axis=1)
        scores = scores_by_class.max(axis=1)
        selected = scores >= CONFIDENCE_THRESHOLD
        predictions, scores, class_ids = predictions[selected], scores[selected], class_ids[selected]
        if not len(predictions):
            return []

        cx, cy, width, height = predictions[:, :4].T
        boxes = np.stack(
            (
                (cx - width / 2 - pad_x) / scale,
                (cy - height / 2 - pad_y) / scale,
                (cx + width / 2 - pad_x) / scale,
                (cy + height / 2 - pad_y) / scale,
            ),
            axis=1,
        )
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h - 1)
        boxes = boxes.astype(np.int32)

        detections = []
        # Apply NMS separately per gesture class so different gestures do not suppress each other.
        for class_id in CLASS_NAMES:
            indices = np.where(class_ids == class_id)[0]
            for local_index in nms(boxes[indices], scores[indices], IOU_THRESHOLD):
                index = indices[local_index]
                detections.append(
                    {"box": boxes[index], "class_id": int(class_id), "score": float(scores[index])}
                )
        return detections


def choose_players(detections, frame_width):
    """Choose the most confident detection from each half of the image."""
    players = [None, None]
    for detection in detections:
        x1, _, x2, _ = detection["box"]
        side = 0 if (x1 + x2) / 2 < frame_width / 2 else 1
        if players[side] is None or detection["score"] > players[side]["score"]:
            players[side] = detection
    return players


def stable_gesture(history):
    values = [value for value in history if value is not None]
    if len(values) < STABLE_FRAMES - 2:
        return None
    gesture, count = Counter(values).most_common(1)[0]
    return gesture if count >= STABLE_FRAMES - 2 else None


def winner(left, right):
    if left == right:
        return DRAW
    # rock beats scissors, paper beats rock, scissors beats paper
    return PLAYER_1_WINS if (left - right) % 3 == 1 else PLAYER_2_WINS


def draw_detection(frame, detection, player_name):
    if detection is None:
        return
    x1, y1, x2, y2 = detection["box"]
    class_id = detection["class_id"]
    color = CLASS_COLORS[class_id]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    label = f"{player_name}: {CLASS_NAMES[class_id]} {detection['score']:.2f}"
    cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def centered_text(frame, text, y, scale=1.0, color=(255, 255, 255), thickness=2):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = max(0, (frame.shape[1] - size[0]) // 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {ENGINE_PATH}")

    detector = RPSDetector(ENGINE_PATH)
    camera = cv2.VideoCapture(CAMERA_ID)
    configure_camera(camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_ID}")

    cv2.namedWindow("Two Player RPS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Two Player RPS", CAMERA_WIDTH, CAMERA_HEIGHT)
    histories = make_histories()
    countdown_end = None
    result = None
    result_until = 0.0
    scores = [0, 0]

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")

            frame = cv2.flip(frame, 1)  # mirror view for natural movement
            detections = detector.detect(frame)
            players = choose_players(detections, frame.shape[1])
            for index, detection in enumerate(players):
                record_detection(histories[index], detection)
                draw_detection(frame, detection, f"PLAYER {index + 1}")

            frame_height, frame_width = frame.shape[:2]

            # Header, two-player divider, and scoreboard.
            cv2.rectangle(frame, (0, 0), (frame_width, 42), (20, 24, 34), -1)
            centered_text(frame, "2 PLAYER ROCK PAPER SCISSORS", 30, 0.75, (255, 255, 255), 2)
            cv2.line(
                frame,
                (frame_width // 2, 42),
                (frame_width // 2, frame_height - 92),
                (0, 255, 255),
                2,
            )
            cv2.rectangle(
                frame,
                (0, frame_height - 92),
                (frame_width, frame_height),
                (20, 24, 34),
                -1,
            )
            now = time.monotonic()
            if countdown_end is not None:
                seconds = max(0, int(np.ceil(countdown_end - now)))
                centered_text(frame, str(seconds), 100, 2.5, (0, 255, 255), 5)
                if now >= countdown_end:
                    left = stable_gesture(histories[0])
                    right = stable_gesture(histories[1])
                    if left is None or right is None:
                        result = "SHOW TWO HANDS - TRY AGAIN"
                    else:
                        round_result = winner(left, right)
                        apply_round_score(scores, round_result)
                        result = f"{CLASS_NAMES[left]} vs {CLASS_NAMES[right]} - {round_result}"
                    result_until = now + 3.0
                    countdown_end = None

            if result and now < result_until:
                centered_text(frame, result, frame_height - 65, 0.68, (0, 255, 255), 2)
            else:
                centered_text(frame, "READY", frame_height - 65, 0.65, (190, 200, 215), 2)

            score_text = f"PLAYER 1   {scores[0]}  :  {scores[1]}   PLAYER 2"
            centered_text(frame, score_text, frame_height - 37, 0.72, (80, 220, 255), 2)
            centered_text(frame, "SPACE: START   R: RESET SCORE   Q: QUIT", frame_height - 12, 0.45)

            cv2.imshow("Two Player RPS", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                scores[:] = [0, 0]
                result = "SCORE RESET"
                result_until = time.monotonic() + 1.5
            if key == ord(" ") and countdown_end is None:
                clear_histories(histories)
                result = None
                countdown_end = time.monotonic() + 3.0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
