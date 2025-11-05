


# Monocular 2D-to-BEV Detection Pipeline

## Project Summary & Results

This project implements a monocular 2D-to-BEV (Bird's Eye View) detection pipeline for nuScenes, using YOLO11n for 2D detection and a ResNet50-based regression model for BEV localization. The pipeline is modular, config-driven, and supports end-to-end training, inference, and visualization. Results show accurate BEV localization for objects within 30m, with increasing error at longer distances due to monocular depth ambiguity.

## Quick Start

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure dataset path in config.yaml

# 3. Train the model
python train.py --config config.yaml

# 4. Run inference on test set
python pipeline.py --config config.yaml --checkpoint outputs/checkpoints/best_model.pth --max-samples 100

# 5. View results
ls outputs/predictions/
cat outputs/predictions/metrics.json
```

## Project Structure

```
mono-bev/
├── data_loader.py         # nuScenes data loading and preprocessing
├── detector.py            # YOLO11n 2D detection
├── bev_transform.py       # 2D-to-BEV transformation logic
├── train.py               # Training loop for BEV regression
├── pipeline.py            # End-to-end inference pipeline
├── visualizer.py          # Visualization utilities
├── config.yaml            # Main configuration file
├── config_low_memory.yaml # Low-memory config
├── requirements.txt       # Python dependencies
├── outputs/               # Checkpoints, logs, predictions
├── monitor_training.sh    # Training monitor script
├── TEST_RESULTS.md        # Quantitative/qualitative results
└── README.md              # Documentation
```

## Configuration & Requirements

- **Python:** 3.8+
- **PyTorch:** 2.4.1
- **nuScenes:** v1.0-trainval (download separately)
- **YOLO11n:** Auto-downloaded by pipeline
- **Config:** All parameters (paths, batch size, learning rate, etc.) are set in `config.yaml`.
- **Hardware:** CUDA GPU recommended (tested on RTX 4060)

## Pipeline Overview

1. **Data Loading:** nuScenes samples loaded, 3D GT projected to 2D/BEV
2. **2D Detection:** YOLO11n detects objects in images, maps COCO to nuScenes classes
3. **BEV Regression:** ResNet50 (frozen) predicts BEV coordinates from 2D detections
4. **Training:** Gradient accumulation, early stopping, checkpointing, logging
5. **Inference:** End-to-end processing, visualizations, metrics computation
6. **Visualization:** 2D overlays, BEV plots, side-by-side comparison

## Technical Details

- **YOLO Detector:** Batch inference, auto-download, COCO→nuScenes mapping
- **BEV Model:** ResNet50 backbone (frozen), multi-head regression (position, orientation, size), IoU-based matching
- **Loss:** Smooth L1 (Huber), weighted sum over all heads
- **Optimizer:** Adam, LR=1e-4, gradient accumulation (8 steps)
- **Evaluation:** Position MAE, orientation error, size error, valid prediction rate
- **Config:** All experiment settings in `config.yaml`

### Model Architecture

```
Input Image (3, 450, 800)
    ↓
YOLO11n Detector
    ↓
2D Detections [N, 6] (x1, y1, x2, y2, conf, class)
    ↓
Crop & Resize (224×224)
    ↓
ResNet50 Encoder (frozen)
    ↓
Feature Vector (2048)
    ↓
┌─────────────┬──────────────┬─────────────┐
│ Position    │ Orientation  │ Size        │
│ Head (2)    │ Head (1)     │ Head (3)    │
└─────────────┴──────────────┴─────────────┘
    ↓
BEV Coordinates (x, y, yaw, length, width, height)
```

## Usage Examples

### Training
```bash
python train.py --config config.yaml
```

### Inference
```bash
python pipeline.py --config config.yaml --checkpoint outputs/checkpoints/best_model.pth --max-samples 10
```

### YOLO Detection Only
```python
from detector import YOLODetector
import yaml, cv2
with open('config.yaml') as f: config = yaml.safe_load(f)
detector = YOLODetector(config)
image = cv2.imread('path/to/image.jpg')
detections = detector.detect(image)
print(f'Detected {len(detections)} objects')
for det in detections:
    print(f"  {det['class']}: {det['confidence']:.2f}")
```

### Analyze Predictions
```python
import json
with open('outputs/predictions/metrics.json') as f:
    metrics = json.load(f)
print(f"Average Position Error: {metrics['position_mae']:.2f}m")
print(f"Samples Processed: {metrics['total_samples']}")
print(f"Valid Predictions: {metrics['valid_predictions']}")
```

## Limitations & Future Work

- Monocular only (no LiDAR/radar, no temporal fusion)
- Single camera view (front only)
- Position error increases >30m
- Frozen backbone (ResNet50)
- No tracking or temporal fusion

**Potential improvements:**
- Unfreeze backbone, multi-camera, temporal fusion, advanced architectures, sensor fusion, tracking

## References & Metadata

- [nuScenes Dataset](https://arxiv.org/abs/1903.11027)
- [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [Lift, Splat, Shoot: BEV from Images](https://arxiv.org/abs/2008.05711)
- [nuScenes Official Website](https://www.nuscenes.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/)

**Last Updated:** November 5, 2025  
**Tested On:** Ubuntu 20.04, CUDA 11.8, RTX 4060 Laptop GPU
