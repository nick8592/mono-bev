"""
Monocular 2D-to-BEV Detection Package

This package implements a monocular 2D-to-BEV (Bird's Eye View) detection pipeline
for autonomous driving scenarios using the nuScenes dataset.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

from src.models.detector import YOLODetector
from src.models.bev_transform import BEVTransformModel, BEVTransformLoss
from src.data.data_loader import NuScenesDataset, create_dataloaders
from src.visualization.visualizer import BEVVisualizer, compute_bev_metrics

__all__ = [
    'YOLODetector',
    'BEVTransformModel',
    'BEVTransformLoss',
    'NuScenesDataset',
    'create_dataloaders',
    'BEVVisualizer',
    'compute_bev_metrics'
]
