"""
Models Module

This module contains neural network models for the 2D-to-BEV pipeline:
- detector: YOLO-based 2D object detection
- bev_transform: ResNet50-based 2D-to-BEV transformation
"""

from .detector import YOLODetector, extract_detection_features
from .bev_transform import (
    BEVTransformModel,
    BEVTransformLoss,
    prepare_batch_for_bev_model,
    prepare_bev_targets,
    compute_iou_2d
)

__all__ = [
    'YOLODetector',
    'extract_detection_features',
    'BEVTransformModel',
    'BEVTransformLoss',
    'prepare_batch_for_bev_model',
    'prepare_bev_targets',
    'compute_iou_2d'
]
