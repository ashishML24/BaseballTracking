# Baseball Speed Vector Estimation

## Overview
This repository implements an end-to-end baseball tracking and speed estimation pipeline using a high-speed monocular image sequence. The system detects a baseball, backtracks it through frames, reconstructs its 3D trajectory using camera geometry and known ball radius, and estimates world-space speed.

The implementation is designed to be modular, physically grounded, and interpretable, with separate stages for detection, 3D reconstruction, filtering, ballistic validation, and visualization.

![Baseball tracking animation](outputs/detections_RefinementWithConstantBallRadius/ball_tracking.gif)

## System Diagram

The pipeline follows this flow:

```
Input image frames
        |
        V
Ball detection + separation frame
        |
        V
Template-based forward/backward tracking
        |
        V
Pixel center + radius estimates per frame
        |
        V
Depth estimation (fixed / per-frame)
        |
        V
Backproject into 3D camera space
        |
        V
Transform 3D positions to world coordinates
        |
        V
Velocity estimation + Kalman filtering
        |
        V
Ballistic fit + confidence intervals
        |
        V
Visualization and result images
```

## Setup Instructions

### 1. Install dependencies

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Prepare input data

Place the input image sequence into `data/`. The default script searches for `data/*.bmp`, but it can be updated for other image extensions.

### 3. Run the pipeline

From the repository root:

```bash
python run_pipeline.py
```

### 4. Configure the pipeline

In `run_pipeline.py`, edit the user input section to choose one of the modes:

- `Ball_refinement = False`, `using_fixed_depth = True`
  - template matching only, fixed depth from separation frame
- `Ball_refinement = True`, `using_fixed_depth = True`
  - template matching plus ball refinement, fixed depth
- `Ball_refinement = True`, `using_fixed_depth = False`
  - template matching plus ball refinement, per-frame depth estimate

### 5. Docker usage

`Commands.txt` contains Docker commands for building and running the container:

```bash
docker build -t baseball-speed .

docker run -it --rm -v /path/to/repo:/app -v /path/to/repo/outputs:/app/outputs baseball-speed /bin/bash
```

Inside the container, run:

```bash
python run_pipeline.py
```

## Output and Sample Results

The pipeline saves outputs under `outputs/detections_xxx`:


## Technology Deck

- Python 3
- OpenCV for image processing and ball detection
- NumPy for numerical operations
- SciPy for ballistic fitting and statistics
- Matplotlib for visualization
- natsort for sorted frame loading
- Docker for containerized execution

## Key Features

- Adaptive template-based ball tracking
- Separation frame detection and template initialization
- Optional ball refinement using radial edge detection and segmentation
- 3D reconstruction using known ball radius and camera geometry
- Kalman filtering of position estimates
- Ballistic model fitting with confidence intervals
- Comprehensive visualization of speed, trajectory, and uncertainty

## Future Improvements

- Add robust fallback detection for frames with low template confidence
- Improve ball segmentation using deep learning or more advanced shape models
- Calibrate the camera intrinsics and extrinsics from real-world data
- Add multi-camera support for stereo / multi-view reconstruction
- Estimate ball spin and launch angle with higher-resolution data
- Add automated result reporting and in-video overlay generation
- Improve uncertainty propagation through the full pipeline

## Notes

- The pipeline assumes a fixed frame rate (`FPS = 240`) and known camera intrinsics.
- `config.py` stores the camera pose, ball radius, and pixel-size constants.
- The code is designed for offline analysis rather than real-time processing.

## Author
Ashish Saxena
