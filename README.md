# Monocular 2D-to-BEV Detection Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, production-ready implementation of a monocular 2D-to-BEV (Bird's Eye View) detection pipeline for autonomous driving using the nuScenes dataset. This project combines YOLO11n for 2D object detection with a ResNet50-based regression model for BEV localization.

## 🎯 Key Features

- **Modular Architecture**: Clean separation of concerns with organized package structure
- **End-to-End Pipeline**: From 2D detection to BEV coordinates prediction
- **Production Ready**: Includes training, inference, visualization, and evaluation
- **Configurable**: YAML-based configuration for easy experimentation
- **Well Documented**: Comprehensive docstrings and documentation
- **Memory Efficient**: Gradient accumulation and optimized data loading

## 📁 Project Structure

```
mono-bev/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default configuration
│   └── low_memory.yaml        # Low-memory configuration
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py    # nuScenes dataset loader
│   ├── models/                # Neural network models
│   │   ├── __init__.py
│   │   ├── detector.py        # YOLO11n detector
│   │   └── bev_transform.py   # ResNet50 BEV model
│   ├── visualization/         # Visualization tools
│   │   ├── __init__.py
│   │   └── visualizer.py      # 2D and BEV plotting
│   └── utils/                 # Utility functions
│       └── __init__.py
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   └── pipeline.py           # Inference pipeline
├── tests/                     # Unit tests
│   └── test_pipeline.py
├── docs/                      # Additional documentation
│   ├── ARCHITECTURE.md       # System architecture
│   └── API.md                # API documentation
├── outputs/                   # Training outputs
│   ├── checkpoints/          # Model checkpoints
│   ├── logs/                 # Training logs
│   └── predictions/          # Inference results
├── yolov11_finetune/         # YOLO fine-tuning (optional)
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── README.md                 # This file
├── CONTRIBUTING.md           # Contribution guidelines
└── LICENSE                   # MIT License
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/nick8592/mono-bev.git
cd mono-bev

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package and dependencies
pip install -e .
# pip install -r requirements.txt  # If needed
```

### 2. Dataset Setup

Download the nuScenes dataset:

```bash
# Download from https://www.nuscenes.org/download
# Place in mono-bev/nuscenes/ directory
```

Update the dataset path in `configs/default.yaml`:

```yaml
data:
  nuscenes_root: "/path/to/nuscenes"
  version: "v1.0-trainval"
```

### 3. Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train with low-memory configuration
python scripts/train.py --config configs/low_memory.yaml

# Or use the installed command
monobev-train --config configs/default.yaml
```

### 4. Inference

```bash
# Run inference on test set
python scripts/pipeline.py \
    --config configs/default.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --max-samples 100

# Or use the installed command
monobev-infer --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth
```

### 5. View Results

```bash
# Check metrics
cat outputs/predictions/metrics.json

# View visualizations
ls outputs/predictions/*.jpg
```

## 🔧 Configuration

The pipeline is highly configurable through YAML files. Key parameters:

```yaml
# Model Architecture
detector:
  model_name: "yolo11n"
  confidence_threshold: 0.5
  device: "cuda"

bev_model:
  input_size: [224, 224]
  pretrained: true
  freeze_backbone: true

# Training
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10

# Visualization
visualization:
  bev_range: [-50, 50, -50, 50]  # [x_min, x_max, y_min, y_max]
  image_format: "jpg"             # jpg or png
```

See `configs/default.yaml` for all available options.

## 📈 Model Architecture

```
Input Image (H, W, 3)
    ↓
┌────────────────────┐
│   YOLO11n Detector │ → 2D Bounding Boxes
└────────────────────┘
    ↓
Crop & Resize (224×224)
    ↓
┌────────────────────┐
│  ResNet50 Encoder  │ (frozen)
│   (2048 features)  │
└────────────────────┘
    ↓
[Features + 2D Box + Camera Intrinsics + Class]
    ↓
┌──────────────┬───────────────┬──────────────┐
│  Position    │  Orientation  │    Size      │
│  Head (x,y,z)│  Head (yaw)   │  Head (w,l,h)│
└──────────────┴───────────────┴──────────────┘
    ↓
BEV Coordinates
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_pipeline.py -v
```

## 🔬 Technical Details

### Dependencies

- **PyTorch** >= 2.0.0: Deep learning framework
- **Ultralytics** >= 8.0.0: YOLO implementation
- **nuScenes-devkit**: Dataset tools
- **OpenCV**: Image processing
- **Matplotlib**: Visualization

### Training Features

- ✅ Gradient accumulation for effective batch size scaling
- ✅ Early stopping with patience
- ✅ Learning rate scheduling
- ✅ Automatic checkpointing
- ✅ Training history logging
- ✅ Validation metrics

### Inference Features

- ✅ Batch processing
- ✅ Automatic visualization generation
- ✅ Metrics computation
- ✅ Side-by-side 2D/BEV comparison plots

## 🎓 Limitations

### Current Limitations

- Monocular only (no LiDAR/radar fusion)
- Single camera view (front camera only)
- Frozen backbone (transfer learning only)
- No temporal fusion across frames
- No object tracking

## 📖 Additional Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Detailed system design
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [nuScenes Dataset](https://arxiv.org/abs/1903.11027)
- [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [Lift, Splat, Shoot: BEV from Images](https://arxiv.org/abs/2008.05711)
- [nuScenes Official Website](https://www.nuscenes.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📧 Contact

For questions or issues, please open an issue on GitHub:
- GitHub: [@nick8592](https://github.com/nick8592)

## 🌟 Citation

If you use this code in your research, please cite:

```bibtex
@software{mono_bev_2025,
  author = {Nick Pai},
  title = {Monocular 2D-to-BEV Detection Pipeline},
  year = {2025},
  url = {https://github.com/nick8592/mono-bev}
}
```

---

**Last Updated:** November 5, 2025  
**Version:** 1.0.0
