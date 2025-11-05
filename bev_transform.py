"""
ResNet50-based 2D-to-BEV Regression Model
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple
import numpy as np


class BEVTransformModel(nn.Module):
    """
    ResNet50-based model for transforming 2D detections to BEV coordinates.
    
    Input: Image crop of detected object + 2D bbox features + camera intrinsics
    Output: BEV coordinates (x, y, z, yaw, width, length, height)
    """
    
    def __init__(self, config: dict):
        """
        Initialize BEV transformation model.
        
        Args:
            config: Configuration dictionary
        """
        super(BEVTransformModel, self).__init__()
        
        self.config = config
        self.num_classes = config['bev_model']['num_classes']
        self.dropout = config['bev_model']['dropout']
        
        # Load pretrained ResNet50 with new API
        if config['bev_model']['pretrained']:
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Optionally freeze backbone
        if config['bev_model']['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension from ResNet50
        feature_dim = self.backbone.fc.in_features
        
        # Remove original FC layer
        self.backbone.fc = nn.Identity()
        
        # Additional features: 2D bbox (4), camera intrinsics (9), class (num_classes)
        # Total additional features: 4 + 9 + num_classes
        self.additional_dim = 4 + 9 + self.num_classes
        
        # Combined feature dimension
        combined_dim = feature_dim + self.additional_dim
        
        # Regression heads for BEV coordinates
        self.position_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 3)  # x, y, z
        )
        
        self.orientation_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 2)  # sin(yaw), cos(yaw) for better regression
        )
        
        self.size_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 3)  # width, length, height
        )
    
    def forward(self, image_crop: torch.Tensor, bbox_2d: torch.Tensor, 
                camera_intrinsic: torch.Tensor, class_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            image_crop: Cropped image patches (B, 3, H, W)
            bbox_2d: 2D bounding boxes (B, 4) - [x1, y1, x2, y2]
            camera_intrinsic: Camera intrinsic matrix (B, 3, 3)
            class_id: One-hot encoded class labels (B, num_classes)
            
        Returns:
            Dictionary with predicted BEV coordinates
        """
        # Extract image features
        image_features = self.backbone(image_crop)  # (B, feature_dim)
        
        # Flatten camera intrinsic
        camera_flat = camera_intrinsic.reshape(camera_intrinsic.size(0), -1)  # (B, 9)
        
        # Concatenate all features
        combined_features = torch.cat([image_features, bbox_2d, camera_flat, class_id], dim=1)
        
        # Predict BEV coordinates
        position = self.position_head(combined_features)  # (B, 3) - x, y, z
        orientation = self.orientation_head(combined_features)  # (B, 2) - sin(yaw), cos(yaw)
        size = self.size_head(combined_features)  # (B, 3) - width, length, height
        
        # Convert orientation to yaw angle
        yaw = torch.atan2(orientation[:, 0], orientation[:, 1])  # (B,)
        
        return {
            'position': position,  # x, y, z
            'yaw': yaw.unsqueeze(1),  # yaw angle
            'size': size,  # width, length, height
            'orientation_vec': orientation  # sin(yaw), cos(yaw) for loss computation
        }


class BEVTransformLoss(nn.Module):
    """
    Combined loss function for BEV transformation.
    """
    
    def __init__(self, config: dict):
        """
        Initialize loss function.
        
        Args:
            config: Configuration dictionary
        """
        super(BEVTransformLoss, self).__init__()
        
        self.position_weight = config['training']['loss_weights']['position']
        self.orientation_weight = config['training']['loss_weights']['orientation']
        self.size_weight = config['training']['loss_weights']['size']
        
        self.position_loss = nn.MSELoss()
        self.orientation_loss = nn.MSELoss()
        self.size_loss = nn.MSELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dictionary of predicted values
            targets: Dictionary of target values
            
        Returns:
            total_loss, loss_dict
        """
        # Position loss (x, y, z)
        loss_pos = self.position_loss(predictions['position'], targets['position'])
        
        # Orientation loss (using sin/cos representation)
        loss_orient = self.orientation_loss(predictions['orientation_vec'], targets['orientation_vec'])
        
        # Size loss (width, length, height)
        loss_size = self.size_loss(predictions['size'], targets['size'])
        
        # Combined loss
        total_loss = (self.position_weight * loss_pos + 
                     self.orientation_weight * loss_orient + 
                     self.size_weight * loss_size)
        
        loss_dict = {
            'total': total_loss.item(),
            'position': loss_pos.item(),
            'orientation': loss_orient.item(),
            'size': loss_size.item()
        }
        
        return total_loss, loss_dict


def prepare_batch_for_bev_model(detections: List[List[Dict]], 
                                images: torch.Tensor,
                                camera_intrinsics: List[np.ndarray],
                                target_size: Tuple[int, int] = (224, 224),
                                num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of detections for BEV model input.
    
    Args:
        detections: List of detection lists (one per image)
        images: Batch of images (B, 3, H, W)
        camera_intrinsics: List of camera intrinsic matrices
        target_size: Target size for image crops
        num_classes: Number of object classes
        
    Returns:
        image_crops, bbox_2d, camera_intrinsic, class_one_hot
    """
    import torchvision.transforms.functional as F
    
    all_crops = []
    all_bboxes = []
    all_intrinsics = []
    all_classes = []
    
    for batch_idx, (dets, img, intrinsic) in enumerate(zip(detections, images, camera_intrinsics)):
        for det in dets:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop detection region from image
            crop = img[:, y1:y2, x1:x2]
            
            # Resize to target size
            crop_resized = F.resize(crop, list(target_size))
            
            all_crops.append(crop_resized)
            all_bboxes.append(torch.tensor(bbox, dtype=torch.float32))
            all_intrinsics.append(torch.tensor(intrinsic, dtype=torch.float32))
            
            # One-hot encode class
            class_one_hot = torch.zeros(num_classes)
            if det['class_id'] >= 0 and det['class_id'] < num_classes:
                class_one_hot[det['class_id']] = 1.0
            all_classes.append(class_one_hot)
    
    if len(all_crops) == 0:
        # Return empty tensors if no detections
        return (torch.empty(0, 3, *target_size),
                torch.empty(0, 4),
                torch.empty(0, 3, 3),
                torch.empty(0, num_classes))
    
    image_crops = torch.stack(all_crops)
    bbox_2d = torch.stack(all_bboxes)
    camera_intrinsic = torch.stack(all_intrinsics)
    class_one_hot = torch.stack(all_classes)
    
    return image_crops, bbox_2d, camera_intrinsic, class_one_hot


def compute_iou_2d(box1, box2):
    """
    Compute IoU between two 2D bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def prepare_bev_targets(bev_coords: List[List[Dict]], 
                       detections: List[List[Dict]],
                       gt_boxes_2d: List[List[Dict]]) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]]]:
    """
    Prepare BEV ground truth targets matching detections.
    Uses IoU matching to align detections with ground truth.
    
    Args:
        bev_coords: List of BEV coordinate lists (one per image)
        detections: List of detection lists (one per image)
        gt_boxes_2d: List of ground truth 2D boxes (one per image)
        
    Returns:
        Tuple of (target tensors dict, list of matched indices)
        matched_indices: List of (batch_idx, detection_idx) for each matched detection
    """
    all_positions = []
    all_orientations = []
    all_sizes = []
    matched_indices = []
    
    total_detections = 0
    total_gt = 0
    matched_count = 0
    debug_printed = False
    
    global_det_idx = 0  # Global detection index across all images in batch
    
    for batch_idx, (bev_list, det_list, gt_list) in enumerate(zip(bev_coords, detections, gt_boxes_2d)):
        if not det_list or not bev_list or not gt_list:
            global_det_idx += len(det_list) if det_list else 0
            continue
        
        total_detections += len(det_list)
        total_gt += len(gt_list)
        
        # Debug: print class names once to see the format
        if not debug_printed and len(det_list) > 0 and len(gt_list) > 0:
            print(f"\n[Class Names Debug]")
            print(f"Detection classes: {[d['class'] for d in det_list[:3]]}")
            print(f"GT classes: {[g['class'] for g in gt_list[:3]]}")
            debug_printed = True
        
        # For each detection, find the best matching ground truth using IoU
        for local_det_idx, det in enumerate(det_list):
            det_bbox = det['bbox']  # [x1, y1, x2, y2]
            det_class = det['class'].lower()
            
            best_match_idx = None
            best_iou = 0.3  # Minimum IoU threshold
            
            # Find ground truth with same class and highest IoU
            for i, gt_box in enumerate(gt_list):
                # Use substring matching for class names
                # e.g., "car" matches "vehicle.car", "pedestrian" matches "human.pedestrian.adult"
                gt_class = gt_box['class'].lower()
                if det_class not in gt_class and gt_class not in det_class:
                    continue
                
                # Compute IoU between detection and ground truth
                gt_bbox = gt_box['bbox']
                iou = compute_iou_2d(det_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx is None or best_match_idx >= len(bev_list):
                # No matching ground truth found, skip this detection
                global_det_idx += 1
                continue
            
            matched_count += 1
            matched_indices.append((batch_idx, global_det_idx))
            best_match = bev_list[best_match_idx]
            
            # Position (x, y, z)
            position = torch.tensor([best_match['x'], best_match['y'], best_match['z']], 
                                   dtype=torch.float32)
            all_positions.append(position)
            
            # Orientation (sin(yaw), cos(yaw))
            yaw = best_match['yaw']
            orientation = torch.tensor([np.sin(yaw), np.cos(yaw)], dtype=torch.float32)
            all_orientations.append(orientation)
            
            # Size (width, length, height)
            size = torch.tensor([best_match['width'], best_match['length'], best_match['height']], 
                               dtype=torch.float32)
            all_sizes.append(size)
            
            global_det_idx += 1
        
    # Debug output every 100 calls
    if np.random.random() < 0.01:  # 1% sampling
        print(f"\n[Matching Debug] Detections: {total_detections}, GT: {total_gt}, Matched: {matched_count}")
    
    if len(all_positions) == 0:
        return {
            'position': torch.empty(0, 3),
            'orientation_vec': torch.empty(0, 2),
            'size': torch.empty(0, 3)
        }, []
    
    targets = {
        'position': torch.stack(all_positions),
        'orientation_vec': torch.stack(all_orientations),
        'size': torch.stack(all_sizes)
    }
    
    return targets, matched_indices
