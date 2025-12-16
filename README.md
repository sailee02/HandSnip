<div align="center">
  <h2 align="center">HandSnip</h2>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-blue"/>
    <img src="https://img.shields.io/badge/OpenCV-4.11-orange"/>
    <img src="https://img.shields.io/badge/MediaPipe-green"/>
    <br>
    Gesture-driven screen snipping and recording tool using hand gestures
  </p>
</div>

## About HandSnip

HandSnip is a touch-less, gesture-driven application that uses your webcam to capture screenshots and record your screen using simple hand gestures. No keyboard or mouse required!

### Features

- 🖐️ **Freeze screen** with open palm gesture
- 🤏 **Select region** by pinching and dragging
- 👍 **Save screenshots** with thumbs up
- 👎 **Cancel** with thumbs down or fist
- 👌 **Record screen** with circle gesture
- 📸 Screenshots saved automatically with timestamps
- 🎥 Full-screen video recording

## Gesture Controls

| Gesture | Action |
|---------|--------|
| 🖐️ **Open Palm** | Freeze the current screen |
| 🤏 **Pinch & Drag** | Select screenshot region (draw rectangle) |
| 👍 **Thumbs Up** | Save screenshot and unfreeze |
| 👎 **Thumbs Down** | Cancel selection and unfreeze |
| ✊ **Fist** | Cancel selection and unfreeze |
| 👌 **Circle** | Start/stop full-screen video recording |

## Installation

### Prerequisites

- Python 3.11 or above
- macOS (tested on macOS with M2 chip)
- Webcam with camera permissions enabled

### Setup

1. Clone the repository:
```sh
git clone https://github.com/sailee02/HandSnip.git
cd HandSnip
```

2. Create a virtual environment:
```sh
python3.11 -m venv .venv311
source .venv311/bin/activate
```

3. Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Usage

### Basic Usage

Run HandSnip with preview window:

```sh
python handsnip.py --preview
```

### Advanced Usage

Run with optimized parameters:

```sh
python handsnip.py \
  --out screenshots \
  --video_out video_recordings \
  --preview \
  --palm_frames 2 \
  --palm_spread 0.22 \
  --pinch_thresh 0.10 \
  --drag_gain 4.0 \
  --edge_extrap_thresh 80 \
  --edge_extrap_step 60 \
  --cam_norm_left 0.15 --cam_norm_right 0.85 \
  --cam_norm_top 0.15 --cam_norm_bottom 0.85
```

### Parameters

- `--out`: Directory for saved screenshots (default: `screenshots/`)
- `--video_out`: Directory for saved video recordings (default: `video_recordings/`)
- `--preview`: Show webcam preview window
- `--palm_frames`: Number of consecutive frames to detect open palm (default: 2)
- `--palm_spread`: Minimum finger spread threshold for open palm (default: 0.22)
- `--pinch_thresh`: Maximum distance between thumb and index for pinch (default: 0.10)
- `--drag_gain`: Sensitivity multiplier for drag movement (default: 4.0)
- `--edge_extrap_thresh`: Pixel threshold for edge extrapolation (default: 80)
- `--edge_extrap_step`: Pixels to extend when near edge (default: 60)
- `--cam_norm_left/right/top/bottom`: Camera normalization bounds (default: 0.15-0.85)

## Workflow

### Screenshot Workflow

1. Make an **open palm** gesture → Screen freezes
2. **Pinch** (thumb touching index) and **drag** to select region
3. Release pinch → Rectangle is locked
4. **Thumbs up** → Screenshot saved to `screenshots/YYYYMMDD_HHMMSS.png`
5. **Thumbs down** or **fist** → Cancel and unfreeze

### Video Recording Workflow

1. Make a **circle** gesture (OK sign) → Recording starts
2. Do whatever you want on screen → Recording continues in background
3. Make **circle** gesture again → Recording stops and saves to `video_recordings/{start}_{end}.mp4`

## Project Structure

```
HandSnip/
├── handsnip.py              # Main application
├── lib/
│   ├── data_loader.py       # Data loading utilities
│   ├── image.py             # Image preprocessing
│   └── resnet_model.py      # 3D ResNet model definition
├── scripts/
│   ├── extract_gif_frames.py    # Extract frames from GIFs
│   ├── split_frames.py          # Split dataset into train/val/test
│   └── train_frames_cnn.py      # Train frame-based CNN model
├── dataset/                 # Training data (not in git)
├── screenshots/             # Saved screenshots (created at runtime)
├── video_recordings/        # Saved recordings (created at runtime)
├── requirements.txt
└── README.md
```

## Troubleshooting

### Camera Not Opening

- Check macOS Privacy & Security settings:
  - **Camera** permission for Terminal/IDE
  - **Screen Recording** permission
  - **Accessibility** permission
- Close other apps using the camera (Zoom, FaceTime, etc.)
- Try running with `--preview` flag to see the webcam window

### Can't Select Full Screen

- Increase `--drag_gain` (e.g., 4.0 → 5.0)
- Adjust camera normalization bounds (`--cam_norm_*`)
- Increase `--edge_extrap_thresh` and `--edge_extrap_step`

### Gestures Not Detecting

- Ensure good lighting
- Keep hand clearly visible in webcam frame
- Adjust thresholds:
  - `--palm_spread` (lower = more sensitive)
  - `--pinch_thresh` (higher = easier to pinch)

## Requirements

See `requirements.txt` for full list. Key dependencies:

- `opencv-python` - Video capture and image processing
- `mediapipe` - Hand landmark detection
- `numpy` - Numerical operations
- `Pillow` - Image manipulation
- `mss` - Screen capture
- `pyautogui` - Mouse control and screenshots

## License

This project is open source and available for personal and educational use.

## Acknowledgments

- MediaPipe for hand tracking
- OpenCV for computer vision utilities
