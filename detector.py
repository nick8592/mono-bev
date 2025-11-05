"""
YOLO-based Object Detector for 2D Detection
"""

import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import cv2
from PIL import Image
import os


class YOLODetector:
    """
    Wrapper class for YOLOv11 object detection.
    """
    
    def __init__(self, config: dict):
        """
        Initialize YOLO detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config['detector']['device']
        self.confidence_threshold = config['detector']['confidence_threshold']
        self.iou_threshold = config['detector']['iou_threshold']
        
        # Load YOLO model
        model_name = config['detector']['model_name']
        print(f"Loading YOLO model: {model_name}")
        
        # Check if model file exists, if not YOLO will download it automatically
        # Just need to provide the model name (e.g., 'yolov11n.pt' or 'yolo11n')
        if not os.path.exists(model_name):
            # Remove .pt extension if present, YOLO will handle download
            model_name_base = model_name.replace('.pt', '')
            print(f"Model file not found. Downloading {model_name_base}...")
            self.model = YOLO(model_name_base)
        else:
            self.model = YOLO(model_name)
        
        self.model.to(self.device)
        
        # nuScenes class mapping to COCO classes (YOLO is trained on COCO)
        self.class_mapping = {
            'car': [2],  # car
            'truck': [7],  # truck
            'bus': [5],  # bus
            'motorcycle': [3],  # motorcycle
            'bicycle': [1],  # bicycle
            'pedestrian': [0],  # person
            'traffic_cone': [],  # no direct mapping
            'barrier': [],  # no direct mapping
            'trailer': [7],  # truck (approximate)
            'construction_vehicle': [7]  # truck (approximate)
        }
        
        # Reverse mapping: COCO class ID to nuScenes class
        self.coco_to_nuscenes = {
            0: 'pedestrian',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run object detection on an image.
        
        Args:
            image: Input image as numpy array (H, W, C) or PIL Image
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection confidence
                - class: class name
                - class_id: class ID
        """
        # Run inference
        results = self.model(image, 
                           conf=self.confidence_threshold,
                           iou=self.iou_threshold,
                           verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Map COCO class to nuScenes class
                if cls_id in self.coco_to_nuscenes:
                    nuscenes_class = self.coco_to_nuscenes[cls_id]
                else:
                    continue  # Skip classes not relevant to nuScenes
                
                detection = {
                    'bbox': box.tolist(),
                    'confidence': conf,
                    'class': nuscenes_class,
                    'class_id': self._get_nuscenes_class_id(nuscenes_class),
                    'coco_class_id': cls_id
                }
                
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Run object detection on a batch of images.
        
        Args:
            images: List of input images (each H, W, 3 in 0-255 range)
            
        Returns:
            List of detection lists (one per image)
        """
        if not images or len(images) == 0:
            return []
        
        # Validate images
        for idx, img in enumerate(images):
            if img is None or img.size == 0:
                print(f"Warning: Empty image at index {idx}, skipping")
                continue
            if len(img.shape) != 3 or img.shape[2] != 3:
                print(f"Warning: Invalid image shape at index {idx}: {img.shape}, expected (H, W, 3)")
                continue
        
        # Filter out invalid images
        valid_images = [img for img in images if img is not None and img.size > 0 and len(img.shape) == 3]
        
        if not valid_images:
            return [[] for _ in images]  # Return empty detections for all images
        
        try:
            # Run batch inference
            results = self.model(valid_images, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               verbose=False)
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            return [[] for _ in images]
        
        batch_detections = []
        
        for result in results:
            detections = []
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                if cls_id in self.coco_to_nuscenes:
                    nuscenes_class = self.coco_to_nuscenes[cls_id]
                else:
                    continue
                
                detection = {
                    'bbox': box.tolist(),
                    'confidence': conf,
                    'class': nuscenes_class,
                    'class_id': self._get_nuscenes_class_id(nuscenes_class),
                    'coco_class_id': cls_id
                }
                
                detections.append(detection)
            
            batch_detections.append(detections)
        
        return batch_detections
    
    def _get_nuscenes_class_id(self, class_name: str) -> int:
        """
        Map nuScenes class names to integer IDs.
        """
        class_mapping = {
            'car': 0,
            'truck': 1,
            'bus': 2,
            'trailer': 3,
            'construction_vehicle': 4,
            'pedestrian': 5,
            'motorcycle': 6,
            'bicycle': 7,
            'traffic_cone': 8,
            'barrier': 9
        }
        
        return class_mapping.get(class_name, -1)
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: Input image as numpy array
            detections: List of detections
            
        Returns:
            Image with drawn detections
        """
        img_vis = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(img_vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_vis


def extract_detection_features(image: np.ndarray, detections: List[Dict], 
                               target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """
    Extract cropped regions from detections for further processing.
    
    Args:
        image: Input image
        detections: List of detections
        target_size: Size to resize crops to
        
    Returns:
        List of cropped and resized image patches
    """
    crops = []
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop detection region
        crop = image[y1:y2, x1:x2]
        
        # Resize to target size
        crop_resized = cv2.resize(crop, target_size)
        
        crops.append(crop_resized)
    
    return crops
