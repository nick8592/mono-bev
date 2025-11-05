"""
nuScenes Data Loader for 2D-to-BEV Detection Pipeline
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional
import os


class NuScenesDataset(Dataset):
    """
    Dataset class for loading nuScenes data with 2D detections and BEV ground truth.
    """
    
    def __init__(self, 
                 nusc: NuScenes,
                 samples: List[str],
                 camera: str = 'CAM_FRONT',
                 transform=None):
        """
        Args:
            nusc: NuScenes dataset object
            samples: List of sample tokens
            camera: Camera sensor name
            transform: Optional transform to apply to images
        """
        self.nusc = nusc
        self.samples = samples
        self.camera = camera
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns a sample with image, detections, and BEV ground truth.
        """
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Get camera data
        camera_token = sample['data'][self.camera]
        cam_data = self.nusc.get('sample_data', camera_token)
        
        # Load image
        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = Image.open(img_path).convert('RGB')
        
        # Store original size for validation
        original_size = image.size
        
        # Get calibration data
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        
        # Get ego pose
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Get annotations (3D bounding boxes)
        boxes_3d, boxes_2d, bev_coords = self._get_annotations(sample_token, camera_token, 
                                                                 camera_intrinsic, cs_record, ego_pose)
        
        # Convert image to tensor
        if self.transform:
            image = self.transform(image)
        else:
            # Convert PIL image to numpy array first
            image_np = np.array(image)
            # Ensure it's uint8 and has correct shape (H, W, 3)
            if len(image_np.shape) != 3 or image_np.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {image_np.shape}")
            # Convert to tensor: (H, W, C) -> (C, H, W) and normalize to [0, 1]
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'image_path': img_path,
            'boxes_2d': boxes_2d,
            'boxes_3d': boxes_3d,
            'bev_coords': bev_coords,
            'camera_intrinsic': camera_intrinsic,
            'sample_token': sample_token,
            'camera_token': camera_token
        }
    
    def _get_annotations(self, sample_token: str, camera_token: str, 
                        camera_intrinsic: np.ndarray, cs_record: dict, 
                        ego_pose: dict) -> Tuple[List, List, List]:
        """
        Get 3D boxes, 2D boxes, and BEV coordinates for all objects in the sample.
        """
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', camera_token)
        
        boxes_3d = []
        boxes_2d = []
        bev_coords = []
        
        # Get all annotation tokens for this sample
        ann_tokens = sample['anns']
        
        for ann_token in ann_tokens:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get 3D box in global coordinates
            box = self.nusc.get_box(ann_token)
            
            # Transform box to ego vehicle coordinates
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)
            
            # Transform box to camera coordinates
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            
            # Check if box is visible in camera
            if not box_in_image(box, camera_intrinsic, (cam_data['width'], cam_data['height']), 
                               vis_level=BoxVisibility.ANY):
                continue
            
            # Get 2D bounding box
            corners_3d = box.corners()
            corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
            
            # Get 2D box bounds
            x_min, y_min = corners_2d.min(axis=1)
            x_max, y_max = corners_2d.max(axis=1)
            
            # Clip to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(cam_data['width'], x_max)
            y_max = min(cam_data['height'], y_max)
            
            # Skip if box is too small
            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                continue
            
            # Get BEV coordinates (x, y in ego frame on ground plane)
            # Transform back to ego frame for BEV
            box_ego = self.nusc.get_box(ann_token)
            box_ego.translate(-np.array(ego_pose['translation']))
            box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
            
            bev_x = box_ego.center[0]  # forward
            bev_y = box_ego.center[1]  # left
            bev_z = box_ego.center[2]  # up (height)
            bev_yaw = box_ego.orientation.yaw_pitch_roll[0]  # rotation around z-axis
            
            boxes_3d.append({
                'center': box.center,
                'size': box.wlh,
                'orientation': box.orientation,
                'class': ann['category_name']
            })
            
            boxes_2d.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'class': ann['category_name'],
                'class_id': self._get_class_id(ann['category_name'])
            })
            
            bev_coords.append({
                'x': bev_x,
                'y': bev_y,
                'z': bev_z,
                'yaw': bev_yaw,
                'width': box_ego.wlh[0],
                'length': box_ego.wlh[1],
                'height': box_ego.wlh[2],
                'class': ann['category_name']
            })
        
        return boxes_3d, boxes_2d, bev_coords
    
    def _get_class_id(self, class_name: str) -> int:
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
        
        # Extract base class name
        for key in class_mapping.keys():
            if key in class_name.lower():
                return class_mapping[key]
        
        return -1  # Unknown class


def create_dataloaders(config: dict, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for nuScenes dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, test_loader
    """
    # Initialize NuScenes
    nusc = NuScenes(version=config['data']['version'], 
                    dataroot=config['data']['nuscenes_root'], 
                    verbose=True)
    
    # Get all samples
    all_samples = [s['token'] for s in nusc.sample]
    
    # Split into train and test
    if config['data']['train_scenes'] is not None:
        # Use specified scenes
        train_scenes = set(config['data']['train_scenes'])
        test_scenes = set(config['data']['test_scenes'])
        
        train_samples = []
        test_samples = []
        
        for sample in nusc.sample:
            scene = nusc.get('scene', sample['scene_token'])
            if scene['name'] in train_scenes:
                train_samples.append(sample['token'])
            elif scene['name'] in test_scenes:
                test_samples.append(sample['token'])
    else:
        # Auto-split
        split_idx = int(len(all_samples) * config['data']['train_split_ratio'])
        np.random.seed(42)
        np.random.shuffle(all_samples)
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
    
    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
    
    # Create datasets
    camera = config['camera']['primary_camera']
    train_dataset = NuScenesDataset(nusc, train_samples, camera=camera)
    test_dataset = NuScenesDataset(nusc, test_samples, camera=camera)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader, nusc


def collate_fn(batch):
    """
    Custom collate function to handle variable-length annotations.
    """
    images = torch.stack([item['image'] for item in batch])
    
    return {
        'images': images,
        'image_paths': [item['image_path'] for item in batch],
        'boxes_2d': [item['boxes_2d'] for item in batch],
        'boxes_3d': [item['boxes_3d'] for item in batch],
        'bev_coords': [item['bev_coords'] for item in batch],
        'camera_intrinsics': [item['camera_intrinsic'] for item in batch],
        'sample_tokens': [item['sample_token'] for item in batch],
        'camera_tokens': [item['camera_token'] for item in batch]
    }
