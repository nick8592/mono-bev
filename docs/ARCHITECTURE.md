# System Architecture

This document provides a detailed overview of the Mono-BEV system architecture.

## Overview

The Monocular 2D-to-BEV Detection Pipeline is a modular system that transforms 2D object detections into Bird's Eye View (BEV) coordinates. The pipeline consists of four main components:

1. **Data Loading**: nuScenes dataset loading and preprocessing
2. **2D Detection**: YOLO-based object detection in images
3. **BEV Transformation**: ResNet50-based regression to BEV space
4. **Visualization**: Plotting and evaluation tools

## Component Architecture

### 1. Data Loading Module (`src/data/`)

#### NuScenesDataset

**Purpose**: Load and preprocess nuScenes samples for training/inference

**Key Functions**:
- Load images and camera calibration data
- Project 3D ground truth to 2D and BEV
- Handle variable-length annotations
- Batch collation for efficient training

**Data Flow**:
```
nuScenes Sample
    ↓
Camera Image + Calibration
    ↓
3D Annotations
    ↓
[2D Boxes, BEV Coords, Image Tensor]
```

**Implementation Details**:
- Uses PyTorch Dataset API
- Custom collate function for batching
- Camera intrinsic matrix handling
- Coordinate transformations (world → ego → camera)

### 2. Detection Module (`src/models/detector.py`)

#### YOLODetector

**Purpose**: Detect objects in 2D images using YOLO11n

**Architecture**:
```
Input Image (H, W, 3)
    ↓
YOLO11n Backbone
    ↓
Detection Heads
    ↓
NMS (Non-Maximum Suppression)
    ↓
[bbox, confidence, class] × N objects
```

**Key Features**:
- Pre-trained on COCO dataset
- Class mapping: COCO → nuScenes
- Batch inference support
- Configurable confidence/IoU thresholds

**Class Mapping**:
| COCO Class | nuScenes Class |
|------------|----------------|
| 0 (person) | pedestrian |
| 1 (bicycle) | bicycle |
| 2 (car) | car |
| 3 (motorcycle) | motorcycle |
| 5 (bus) | bus |
| 7 (truck) | truck |

### 3. BEV Transformation Module (`src/models/bev_transform.py`)

#### BEVTransformModel

**Purpose**: Predict BEV coordinates from 2D detections

**Architecture**:

```
Cropped Detection (3, 224, 224)
    ↓
ResNet50 Encoder (frozen)
    ↓
Feature Vector (2048)
    ↓
Concatenate:
  - Image features (2048)
  - 2D bbox (4)
  - Camera intrinsic (9)
  - Class one-hot (10)
    ↓
Combined Features (2071)
    ↓
┌─────────────────┬──────────────────┬────────────────┐
│ Position Head   │ Orientation Head │ Size Head      │
│ FC→ReLU→Dropout │ FC→ReLU→Dropout  │ FC→ReLU→Dropout│
│ 2071→512→256→3  │ 2071→256→128→2   │ 2071→256→128→3 │
└─────────────────┴──────────────────┴────────────────┘
    ↓
[x, y, z, yaw, width, length, height]
```

**Design Decisions**:

1. **Frozen Backbone**: 
   - Reduces memory usage
   - Leverages ImageNet features
   - Focus learning on regression heads

2. **Multi-Head Architecture**:
   - Separate heads for position, orientation, size
   - Allows task-specific learning
   - Easier to tune loss weights

3. **Orientation Representation**:
   - Uses (sin(yaw), cos(yaw)) instead of raw angle
   - Avoids discontinuity at ±π
   - Better regression targets

4. **Input Features**:
   - Image crop: Visual features
   - 2D bbox: Spatial context
   - Camera intrinsic: Projection geometry
   - Class: Object type information

#### BEVTransformLoss

**Loss Function**:
```python
L_total = w_pos * L_position + w_orient * L_orientation + w_size * L_size
```

Where:
- `L_position`: MSE on (x, y, z)
- `L_orientation`: MSE on (sin(yaw), cos(yaw))
- `L_size`: MSE on (width, length, height)

Default weights: `w_pos=1.0, w_orient=0.5, w_size=0.3`

### 4. Visualization Module (`src/visualization/visualizer.py`)

#### BEVVisualizer

**Purpose**: Create visualizations for analysis and evaluation

**Capabilities**:

1. **2D Detection Visualization**:
   - Draw bounding boxes on images
   - Show ground truth vs predictions
   - Class labels and confidence scores

2. **BEV Space Visualization**:
   - Plot objects in bird's eye view
   - Show orientation arrows
   - Ego vehicle reference
   - Configurable range and grid

3. **Comparison Plots**:
   - Side-by-side 2D and BEV views
   - Overlay GT and predictions
   - Save to various formats (JPG/PNG)

**Coordinate System**:
```
BEV Space (visualized):
    ↑ X (forward)
    │
    │
    └──→ -Y (left becomes right)
    
Original:
X (forward), Y (left), Z (up)

Rotation for visualization: 90° CCW
(x, y) → (-y, x)
```

## Training Pipeline

### Training Loop Architecture

```
Epoch Start
    ↓
For each batch:
    │
    ├─→ Load images and GT
    ├─→ Run YOLO detection
    ├─→ Match detections to GT (IoU-based)
    ├─→ Prepare BEV model inputs
    ├─→ Forward pass
    ├─→ Compute loss
    ├─→ Backward pass (with gradient accumulation)
    └─→ Update weights
    ↓
Validation
    ↓
Checkpointing & Early Stopping
    ↓
Next Epoch / End
```

### Gradient Accumulation

To handle limited GPU memory:

```python
accumulation_steps = 8
for batch in dataloader:
    loss = forward_backward(batch)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch size = `batch_size × accumulation_steps`

### IoU-Based Matching

Match detections to ground truth:

1. For each detection, compute IoU with all GT boxes
2. Match to GT with highest IoU (if > 0.3 threshold)
3. Also check class compatibility
4. Only matched detections used for training

This ensures we train on correct detection-GT pairs.

## Inference Pipeline

### End-to-End Flow

```
Input: nuScenes sample
    ↓
Load image + calibration
    ↓
YOLO Detection
    ↓
For each detection:
    ├─→ Crop detection region
    ├─→ Prepare features
    ├─→ BEV model forward
    └─→ BEV coordinates
    ↓
Post-processing & visualization
    ↓
Metrics computation
    ↓
Save results
```

### Batch Processing

The pipeline supports batch processing for efficiency:

```python
# Process multiple samples
for sample_batch in samples:
    images = load_batch(sample_batch)
    detections = detector.detect_batch(images)
    bev_predictions = bev_model.predict_batch(detections)
    save_visualizations(bev_predictions)
```

## Configuration System

### YAML Configuration

The system uses hierarchical YAML configuration:

```yaml
data:           # Dataset parameters
detector:       # YOLO settings
bev_model:      # BEV model architecture
training:       # Training hyperparameters
visualization:  # Plotting options
logging:        # Checkpointing and logs
inference:      # Inference settings
camera:         # Camera configuration
```

Benefits:
- Easy experimentation
- Reproducibility
- No code changes needed
- Version control friendly

### Configuration Inheritance

Multiple configs supported:
- `configs/default.yaml`: Standard settings
- `configs/low_memory.yaml`: For limited GPU memory

Override from command line:
```bash
python scripts/train.py --config configs/custom.yaml
```

## Memory Management

### Optimization Strategies

1. **Frozen Backbone**: Reduces memory by ~40%
2. **Gradient Accumulation**: Smaller batch size, same effective size
3. **Batch Size**: Configurable (default: 8)
4. **GPU Cache Clearing**: Periodic `torch.cuda.empty_cache()`
5. **Reduced Workers**: num_workers=2 for data loading

### Memory Profile

Typical GPU memory usage (RTX 4060):

| Component | Memory |
|-----------|--------|
| ResNet50 (frozen) | ~200MB |
| YOLO11n | ~12MB |
| Batch (8 images) | ~500MB |
| Optimizer states | ~100MB |
| **Total** | **~1GB** |

## Error Handling

### Robustness Features

1. **Empty Detections**: Skip batch, continue training
2. **No Matches**: Skip if no detection-GT pairs
3. **Invalid Images**: Validate shape and type
4. **Missing Files**: Check existence before loading
5. **GPU OOM**: Reduce batch size automatically

### Logging

Comprehensive logging:
- Training progress (loss, metrics)
- Validation results
- Checkpoint saving
- Warning for edge cases

## Performance Considerations

### Bottlenecks

1. **YOLO Inference**: ~50ms per image
2. **BEV Model**: ~10ms per detection
3. **Data Loading**: ~20ms per batch
4. **Visualization**: ~100ms per sample

### Optimization Opportunities

- [ ] TensorRT for YOLO (3-5× speedup)
- [ ] Batch BEV inference (2× speedup)
- [ ] Cached data loading
- [ ] Multi-GPU training
- [ ] Mixed precision (FP16)

## Testing Strategy

### Unit Tests

- Model initialization
- Data loading
- Coordinate transformations
- Visualization functions

### Integration Tests

- End-to-end pipeline
- Training loop
- Inference pipeline

### Validation Tests

- Output shape validation
- Range checking
- Metric computation

## Deployment Considerations

### Production Checklist

- [ ] Model quantization (INT8)
- [ ] ONNX export
- [ ] TensorRT conversion
- [ ] API server (FastAPI/Flask)
- [ ] Docker containerization
- [ ] Monitoring and logging
- [ ] Error recovery

### Real-Time Performance

For real-time deployment:
- Target: 30 FPS (33ms per frame)
- Current: ~5-10 FPS
- Optimization needed: 3-6× speedup

## Future Architecture Improvements

1. **Multi-Camera Fusion**: Combine multiple views
2. **Temporal Model**: Use video sequences
3. **Attention Mechanism**: Focus on important regions
4. **End-to-End Training**: Unfreeze backbone
5. **Depth Estimation**: Explicit depth prediction
6. **Uncertainty Estimation**: Confidence scores

---

Last Updated: November 5, 2025
