# Traffic Light Detection (OpenCV + HSV)

Real-time detection of traffic lights (Red/Yellow/Green) from webcam or a video file using OpenCV. The detector applies HSV color segmentation with adaptive preprocessing and draws labeled boxes with confidence scores. A minimal UI is included to select sources and optionally save processed output.

## Features

- **Automatic color detection**: Robust HSV ranges for Red, Yellow, Green
- **Adaptive preprocessing**: Gray-world white balance, auto gamma, CLAHE
- **Noise handling**: Morphology, contour filters, brightness/saturation gates
- **Tracking and smoothing**: Simple IoU-based track management with lock-on
- **Heuristics**: Basic separation of traffic lights vs vehicle lights
- **UI launcher**: Simple OpenCV-based menu for webcam/video selection
- **Optional recording**: Save annotated output to MP4 with timestamp

## Quick Start

### 1) Run via the UI (recommended)

```bash
./run_detector.sh
```

The script will create a virtual environment (../.venv), install dependencies from `requirements.txt`, and launch the UI. In the UI:

- Press `1` to use a webcam (choose numeric ID, usually `0`)
- Press `2` to provide a video file path
- Choose whether to save the processed output
- Press `q` in the video window to quit; press `d` to toggle debug masks

### 2) Run directly from the CLI

```bash
python traffic_light_detector.py --source 0 --output
```

- **--source / -s**: Camera index like `0` or a video file path
- **--output / -o**: When present, saves the processed video to an MP4 file

Example with a file:

```bash
python traffic_light_detector.py -s "/path/to/video.mp4" -o
```

## Installation

If you prefer manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`):

- opencv-contrib-python (≥ 4.5.0)
- numpy (≥ 1.20.0)

## Controls & Output

- **Keys**: `q` quits, `d` toggles color mask debug panel
- **Boxes/labels**: Color name, type guess, track id, and confidence
- **Saved video**: If enabled, file name like `auto_traffic_light_detection_YYYYMMDD_HHMMSS.mp4`

## Tips & Troubleshooting

- **No camera feed / black window**: Check the `--source` index; try `0` or another number. Ensure another app isn’t using the camera.
- **Could not open video source**: The path may be wrong or unsupported. Try absolute paths and common formats (MP4, AVI, MOV, MKV).
- **Low detections**: Try brighter footage, point at larger signals, or press `d` to view masks and verify colors are captured.
- **Performance**: Reduce input resolution or close other apps. CPU-only processing; no GPU is required.

## Project Structure

- `traffic_light_detector.py`: Core detector and CLI
- `simple_ui.py`: OpenCV-based menu to pick source and save option
- `run_detector.sh`: Helper script to set up venv, install deps, and launch UI
- `requirements.txt`: Minimal dependencies

## How it works (high level)

1. Preprocess each frame: white-balance, gamma correction, blur, HSV+CLAHE
2. Build color masks per class and clean with morphology/median filtering
3. Find candidate contours and filter by size, aspect ratio, circularity, and brightness
4. Score detections (geometry, purity, brightness) and apply NMS
5. Track across frames, smooth boxes/confidence, and label majority color

## License

MIT (or project’s original license). If you use this work, consider crediting the repository and authors.

## Acknowledgements

Built with OpenCV and NumPy. Thanks to the open-source community for tools and prior art on HSV-based traffic light detection.
