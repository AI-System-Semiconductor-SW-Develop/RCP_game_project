# RCP_game_project

Jetson Orin Nano와 YOLO/TensorRT를 사용하는 2인용 가위바위보 게임입니다. 카메라 화면의 왼쪽 손은 Player 1, 오른쪽 손은 Player 2로 인식합니다.

## 실행

Jetson 터미널에서 프로젝트 디렉터리로 이동한 뒤 실행합니다.

```bash
cd RCP_game_project
python3 rps_game.py
```

PyQt5 UI 버전은 다음과 같이 실행합니다.

```bash
python3 rps_game_gui.py
```

XLaunch를 사용할 때는 Windows에서 XLaunch를 먼저 실행하고, WSL/SSH 터미널에서 `DISPLAY`를 설정한 뒤 위 명령을 실행합니다. `<WINDOWS_IP>`에는 Windows 호스트 IP를 입력합니다.

```bash
export DISPLAY=<WINDOWS_IP>:0.0
python3 rps_game.py
```

- `Space`: 3초 카운트다운 후 두 손의 가위바위보 판정
- `Q` 또는 `Esc`: 종료

두 손을 화면 중앙선의 서로 다른 쪽에 놓고, 카운트다운 동안 같은 동작을 유지하세요. 화면은 거울처럼 좌우 반전되어 표시됩니다.

## 필요 파일 및 패키지

- `rps_game_yolo11n_fp16.engine` (현재 Jetson/TensorRT 버전에서 생성한 엔진)
- `opencv-python`, `numpy`, `tensorrt`, `pycuda`

엔진이 없거나 JetPack/TensorRT 버전이 달라 로딩되지 않으면 해당 Jetson에서 ONNX 파일을 다시 변환합니다.

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=rps_game_yolo11n.onnx \
  --saveEngine=rps_game_yolo11n_fp16.engine \
  --fp16
```
