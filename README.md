
# nuScenes 2D-to-BEV Detection Pipeline

## Project Objective
Build a pipeline that detects objects in 2D nuScenes images and projects them into BEV (Bird's Eye View) space using YOLOv11 for detection and a ResNet50 regression model for 2D-to-BEV transformation.

## Input / Output
- **Input:** Camera images from the nuScenes dataset
- **Output:**
  - BEV plots visualizing detected objects
  - Detection results (bounding boxes, classes, BEV coordinates)

## Example Results
*Example output images and detection results will be added here.*

## Limitations / Assumptions
 Only camera images are supported (no LiDAR/radar)
 No object tracking or temporal fusion
 Only YOLOv11 and ResNet50 models implemented

- OpenCV
- nuScenes devkit
- YOLO (object detection backbone)
- ResNet50 (2D-to-BEV regression)
- Matplotlib (visualization)

## Dataset Preparation
- Download the nuScenes dataset from the [official website](https://www.nuscenes.org/download).
- Follow the nuScenes devkit instructions for dataset setup.
- Ensure camera calibration files are available for 2D-to-BEV transformation.


## Pipeline Overview
1. **nuScenes Data Loader**
   - Load 2D images and calibration data from nuScenes.
   - Prepare image batches for inference.
2. **Object Detection (YOLO Backbone)**
   - Use a pretrained YOLO model for 2D object detection.
   - Output bounding boxes and class labels.
3. **2D-to-BEV Regression (ResNet50)**
   - Use a ResNet50-based regression model to transform 2D detections to BEV coordinates.
   - Requires camera intrinsic/extrinsic parameters.
4. **BEV Visualization**
   - Plot detected objects in BEV space using Matplotlib.
   - Visualize results for qualitative analysis.
5. **Modular Code Structure**
   - Organize code for easy expansion:
     - `data_loader.py`: nuScenes data loading
     - `detector.py`: YOLO detection
     - `bev_transform.py`: 2D-to-BEV regression (ResNet50)
     - `visualizer.py`: BEV visualization

## Visualization Tools
To support analysis and debugging, the pipeline includes visualization tools for:
- **Ground Truth BEV:** Plot ground truth object positions in BEV space using Matplotlib or similar libraries.
- **Predicted BEV:** Visualize predicted BEV positions from the regression model alongside ground truth for comparison.
- **YOLO Detection Results:** Display 2D detection results (bounding boxes and classes) on the original nuScenes images.
These visualizations help qualitatively assess detection and regression performance, and support model development and evaluation.

## Obtaining Ground Truth in BEV Space
To get ground truth object positions in BEV space from nuScenes:
1. **3D Bounding Boxes:** Use nuScenes 3D bounding box annotations, which provide object center (x, y, z), size, orientation (yaw), and category in the global coordinate frame.
2. **Sensor Calibration:** Use camera and LiDAR calibration files (intrinsic/extrinsic parameters) to transform between sensor, ego vehicle, and global coordinates.
3. **Ego Vehicle Pose:** Use the ego vehicle’s position and orientation for each sample to accurately map objects to BEV.
4. **Projection to BEV:** Project the 3D bounding box center (x, y) and orientation (yaw) onto the ground plane (BEV) using ego pose and calibration data.
This method allows you to visualize and evaluate detection results against ground truth in BEV space.

## Train/Test Split and Model Evaluation
To train the 2D-to-BEV regression model (ResNet50):
1. **Train/Test Split:**
   - Split the nuScenes dataset into training and testing sets (e.g., by scene).
   - Use the training set to fit the regression model and the test set for evaluation.
2. **Training:**
   - Use 2D detections (YOLO) and corresponding ground truth BEV positions (from 3D bounding boxes) as input/output pairs.
   - Train the regression model to predict BEV coordinates from 2D detections.
3. **Evaluation:**
   - Evaluate model performance on the test set using metrics such as mean squared error (MSE) for BEV position prediction, and detection metrics (precision, recall, mAP) for overall pipeline.
   - Visualize predicted vs. ground truth BEV positions for qualitative analysis.


## Configuration and Logging

### Config File
Training and testing scripts are controlled via a configuration file (e.g., `config.yaml`). This file allows you to specify parameters such as data paths, model weights, batch size, learning rate, and other options for reproducible experiments.

### Logging
Each training run generates logs that record key metrics, configuration settings, and results. Logs are saved for later analysis and reproducibility.

## How to Run
```bash
# Example usage
python pipeline.py --config config.yaml
```

## Future Expansion
- Support for multiple sensors (LiDAR, radar)
- Advanced detection models
- Tracking and temporal fusion

## References
- [nuScenes Dataset](https://www.nuscenes.org/)
- [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [ResNet: Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [BEV Visualization](https://arxiv.org/abs/2007.13716)

---

*Draft updated: November 4, 2025*
