"""PyQt5 front end for the TensorRT rock-paper-scissors game."""

from collections import deque
import os
import random
import sys
import time

import cv2
from PyQt5.QtCore import QLibraryInfo, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from rps_game import (
    CAMERA_HEIGHT,
    CAMERA_ID,
    CAMERA_WIDTH,
    CLASS_NAMES,
    ENGINE_PATH,
    RPSDetector,
    STABLE_FRAMES,
    choose_players,
    draw_detection,
    stable_gesture,
    winner,
)
from ui.ui_rsp_game import Ui_MainWindow


def configure_qt_plugins():
    """Undo the Qt plugin path injected by the non-headless OpenCV wheel."""
    plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

    # OpenCV's bundled font path belongs to its own Qt build as well.
    font_path = os.environ.get("QT_QPA_FONTDIR", "")
    if "/cv2/qt/" in font_path:
        os.environ.pop("QT_QPA_FONTDIR", None)


class RPSGameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("가위바위보 게임")
        self.setFixedSize(640, 480)

        self.detector = None
        self.camera = None
        self.mode = None
        self.countdown_end = None
        self.next_round_at = None
        self.auto_running = False
        self.histories = [
            deque(maxlen=STABLE_FRAMES),
            deque(maxlen=STABLE_FRAMES),
        ]
        self.single_score = [0, 0]  # computer, player
        self.multi_score = [0, 0]  # player 1, player 2

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_frame)

        self.ui.btn_single.clicked.connect(self.show_single_page)
        self.ui.btn_multi.clicked.connect(self.show_multi_page)
        self.ui.single_start.clicked.connect(self.start_round)
        self.ui.btn_start.clicked.connect(self.start_round)
        self.ui.single_stop.clicked.connect(self.stop_game)
        self.ui.btn_stop.clicked.connect(self.stop_game)

        self.ui.lbl_camera.setAlignment(Qt.AlignCenter)
        self.ui.lbl_single_camera.setAlignment(Qt.AlignCenter)
        self.show_menu_page()
        self.update_score_labels()

    def show_menu_page(self):
        self.auto_running = False
        self.next_round_at = None
        self.stop_camera()
        self.mode = None
        self.countdown_end = None
        self.set_start_buttons_enabled(True)
        self.ui.Page.setCurrentWidget(self.ui.menuPage)

    def show_single_page(self):
        self.mode = "single"
        self.ui.Page.setCurrentWidget(self.ui.singlePage)
        self.ui.result_single.setText("GAME START를 누르세요")
        self.start_camera()

    def show_multi_page(self):
        self.mode = "multi"
        self.ui.Page.setCurrentWidget(self.ui.multiPage)
        self.ui.resultLabel.setText("두 손을 화면 좌우에 놓으세요")
        self.start_camera()

    def start_camera(self):
        try:
            if self.detector is None:
                self.detector = RPSDetector(ENGINE_PATH)
            if self.camera is None:
                self.camera = cv2.VideoCapture(CAMERA_ID)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.camera.isOpened():
                raise RuntimeError(f"카메라 {CAMERA_ID}번을 열 수 없습니다.")
            self.timer.start()
        except Exception as error:
            self.stop_camera()
            QMessageBox.critical(self, "실행 오류", str(error))
            self.ui.Page.setCurrentWidget(self.ui.menuPage)

    def stop_camera(self):
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def start_round(self):
        if self.mode is None or self.auto_running:
            return
        self.auto_running = True
        self.set_start_buttons_enabled(False)
        self.begin_countdown()

    def begin_countdown(self):
        for history in self.histories:
            history.clear()
        self.countdown_end = time.monotonic() + 3.0
        self.next_round_at = None

    def stop_game(self):
        """Stop automatic rounds and return to the mode selection page."""
        self.auto_running = False
        self.countdown_end = None
        self.next_round_at = None
        self.show_menu_page()

    def set_start_buttons_enabled(self, enabled):
        self.ui.single_start.setEnabled(enabled)
        self.ui.btn_start.setEnabled(enabled)

    def update_frame(self):
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.show_menu_page()
            QMessageBox.warning(self, "카메라 오류", "카메라 영상을 읽지 못했습니다.")
            return

        frame = cv2.flip(frame, 1)
        try:
            detections = self.detector.detect(frame)
        except Exception as error:
            self.show_menu_page()
            QMessageBox.critical(self, "TensorRT 오류", str(error))
            return

        if self.mode == "single":
            self.process_single_frame(frame, detections)
            target = self.ui.lbl_single_camera
        elif self.mode == "multi":
            self.process_multi_frame(frame, detections)
            target = self.ui.lbl_camera
        else:
            return

        self.update_countdown()
        self.show_frame(frame, target)

    def process_single_frame(self, frame, detections):
        human = max(detections, key=lambda item: item["score"], default=None)
        self.histories[0].append(None if human is None else human["class_id"])
        draw_detection(frame, human, "PLAYER")

    def process_multi_frame(self, frame, detections):
        players = choose_players(detections, frame.shape[1])
        for index, detection in enumerate(players):
            self.histories[index].append(None if detection is None else detection["class_id"])
            draw_detection(frame, detection, f"PLAYER {index + 1}")
        cv2.line(
            frame,
            (frame.shape[1] // 2, 0),
            (frame.shape[1] // 2, frame.shape[0]),
            (0, 255, 255),
            2,
        )

    def update_countdown(self):
        if self.countdown_end is None:
            if (
                self.auto_running
                and self.next_round_at is not None
                and time.monotonic() >= self.next_round_at
            ):
                self.begin_countdown()
            return
        remaining = self.countdown_end - time.monotonic()
        result_label = self.ui.result_single if self.mode == "single" else self.ui.resultLabel
        if remaining > 0:
            result_label.setText(f"{max(1, int(remaining) + 1)}")
            return

        self.countdown_end = None
        if self.mode == "single":
            self.finish_single_round()
        else:
            self.finish_multi_round()
        # Keep the result visible briefly, then automatically start the next round.
        if self.auto_running:
            self.next_round_at = time.monotonic() + 1.5

    def finish_single_round(self):
        human = stable_gesture(self.histories[0])
        if human is None:
            self.ui.result_single.setText("손을 인식하지 못했습니다. 다시 시도하세요.")
            return

        computer = random.choice(tuple(CLASS_NAMES))
        round_result = winner(computer, human)
        if round_result == "PLAYER 1 WINS":
            self.single_score[0] += 1
            result_text = "컴퓨터 승리"
        elif round_result == "PLAYER 2 WINS":
            self.single_score[1] += 1
            result_text = "플레이어 승리"
        else:
            result_text = "무승부"
        self.ui.result_single.setText(
            f"컴퓨터 {CLASS_NAMES[computer]} : {CLASS_NAMES[human]} 플레이어 - {result_text}"
        )
        self.update_score_labels()

    def finish_multi_round(self):
        player1 = stable_gesture(self.histories[0])
        player2 = stable_gesture(self.histories[1])
        if player1 is None or player2 is None:
            self.ui.resultLabel.setText("두 손을 모두 인식하지 못했습니다.")
            return

        round_result = winner(player1, player2)
        if round_result == "PLAYER 1 WINS":
            self.multi_score[0] += 1
            result_text = "PLAYER 1 승리"
        elif round_result == "PLAYER 2 WINS":
            self.multi_score[1] += 1
            result_text = "PLAYER 2 승리"
        else:
            result_text = "무승부"
        self.ui.resultLabel.setText(
            f"{CLASS_NAMES[player1]} : {CLASS_NAMES[player2]} - {result_text}"
        )
        self.update_score_labels()

    def update_score_labels(self):
        self.ui.score_single.setText(
            f"COMPUTER {self.single_score[0]} : {self.single_score[1]} PLAYER"
        )
        self.ui.scoreLabel.setText(
            f"PLAYER 1 {self.multi_score[0]} : {self.multi_score[1]} PLAYER 2"
        )

    @staticmethod
    def show_frame(frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


def main():
    configure_qt_plugins()
    app = QApplication(sys.argv)
    window = RPSGameWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
