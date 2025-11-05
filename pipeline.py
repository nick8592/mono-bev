"""
Main Pipeline for 2D-to-BEV Object Detection
"""

import torch
import numpy as np
import yaml
import os
import argparse
from tqdm import tqdm
from PIL import Image

from data_loader import NuScenesDataset
from nuscenes.nuscenes import NuScenes
from detector import YOLODetector
from bev_transform import BEVTransformModel, prepare_batch_for_bev_model
from visualizer import BEVVisualizer, compute_bev_metrics


class BEVPipeline:
    """
    End-to-end pipeline for 2D-to-BEV object detection.
    """
    
    def __init__(self, config: dict, checkpoint_path: str = None):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to trained BEV model checkpoint
        """
        self.config = config
        self.device = torch.device(config['detector']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize detector
        print("Loading YOLO detector...")
        self.detector = YOLODetector(config)
        
        # Initialize BEV model
        print("Loading BEV transformation model...")
        self.bev_model = BEVTransformModel(config).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.bev_model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully!")
        else:
            print("Warning: No checkpoint provided or file not found. Using untrained model.")
        
        self.bev_model.eval()
        
        # Initialize visualizer
        self.visualizer = BEVVisualizer(config)
        
        # Create output directory
        self.output_dir = config['inference']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_image(self, image: np.ndarray, camera_intrinsic: np.ndarray,
                     gt_bev: list = None, sample_token: str = None) -> dict:
        """
        Process a single image through the pipeline.
        
        Args:
            image: Input image (H, W, 3)
            camera_intrinsic: Camera intrinsic matrix (3, 3)
            gt_bev: Ground truth BEV coordinates (optional)
            sample_token: Sample identifier (optional)
            
        Returns:
            Dictionary with detections and BEV predictions
        """
        # Run 2D detection
        detections = self.detector.detect(image)
        
        if len(detections) == 0:
            print("No objects detected in image")
            return {
                'detections_2d': [],
                'predictions_bev': [],
                'gt_bev': gt_bev if gt_bev else []
            }
        
        print(f"Detected {len(detections)} objects")
        
        # Prepare input for BEV model
        # Convert image to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        image_crops, bbox_2d, cam_intrinsic_batch, class_one_hot = prepare_batch_for_bev_model(
            [detections], image_tensor, [camera_intrinsic],
            target_size=tuple(self.config['bev_model']['input_size']),
            num_classes=self.config['bev_model']['num_classes']
        )
        
        if len(image_crops) == 0:
            return {
                'detections_2d': detections,
                'predictions_bev': [],
                'gt_bev': gt_bev if gt_bev else []
            }
        
        # Move to device
        image_crops = image_crops.to(self.device)
        bbox_2d = bbox_2d.to(self.device)
        cam_intrinsic_batch = cam_intrinsic_batch.to(self.device)
        class_one_hot = class_one_hot.to(self.device)
        
        # Run BEV prediction
        with torch.no_grad():
            predictions = self.bev_model(image_crops, bbox_2d, cam_intrinsic_batch, class_one_hot)
        
        # Convert predictions to BEV format
        predictions_bev = []
        for i in range(len(detections)):
            pred_bev = {
                'x': predictions['position'][i, 0].item(),
                'y': predictions['position'][i, 1].item(),
                'z': predictions['position'][i, 2].item(),
                'yaw': predictions['yaw'][i, 0].item(),
                'width': predictions['size'][i, 0].item(),
                'length': predictions['size'][i, 1].item(),
                'height': predictions['size'][i, 2].item(),
                'class': detections[i]['class'],
                'confidence': detections[i]['confidence']
            }
            predictions_bev.append(pred_bev)
        
        # Compute metrics if ground truth is available
        metrics = None
        if gt_bev:
            metrics = compute_bev_metrics(gt_bev, predictions_bev)
            print(f"Metrics: Position MAE={metrics['position_mae']:.2f}m, "
                  f"Orientation Error={metrics['orientation_error']:.2f}rad")
        
        return {
            'detections_2d': detections,
            'predictions_bev': predictions_bev,
            'gt_bev': gt_bev if gt_bev else [],
            'metrics': metrics
        }
    
    def run_on_dataset(self, nusc: NuScenes, samples: list = None, max_samples: int = None):
        """
        Run pipeline on nuScenes dataset.
        
        Args:
            nusc: NuScenes dataset object
            samples: List of sample tokens (if None, use all)
            max_samples: Maximum number of samples to process
        """
        if samples is None:
            samples = [s['token'] for s in nusc.sample]
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"\nProcessing {len(samples)} samples...")
        
        camera = self.config['camera']['primary_camera']
        all_metrics = []
        
        for sample_token in tqdm(samples, desc="Processing samples"):
            sample = nusc.get('sample', sample_token)
            
            # Get camera data
            camera_token = sample['data'][camera]
            cam_data = nusc.get('sample_data', camera_token)
            
            # Load image
            img_path = os.path.join(nusc.dataroot, cam_data['filename'])
            image = np.array(Image.open(img_path).convert('RGB'))
            
            # Get calibration
            cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            camera_intrinsic = np.array(cs_record['camera_intrinsic'])
            
            # Get ground truth BEV (simplified - use dataset loader for proper transformation)
            dataset = NuScenesDataset(nusc, [sample_token], camera=camera)
            sample_data = dataset[0]
            gt_bev = sample_data['bev_coords']
            
            # Process image
            results = self.process_image(image, camera_intrinsic, gt_bev, sample_token)
            
            # Save visualizations
            if self.config['inference']['save_visualizations']:
                # Get image format from config
                img_format = self.config['visualization'].get('image_format', 'png')
                
                # Get 2D detection visualization settings
                det_2d_config = self.config['visualization'].get('detection_2d', {})
                show_gt_2d = det_2d_config.get('show_ground_truth', True)
                show_pred_2d = det_2d_config.get('show_predictions', True)
                show_labels_2d = det_2d_config.get('show_class_labels', True)
                
                # Get BEV visualization settings
                bev_config = self.config['visualization'].get('bev', {})
                show_gt_bev = bev_config.get('show_ground_truth', True)
                show_pred_bev = bev_config.get('show_predictions', True)
                show_labels_bev = bev_config.get('show_class_labels', True)
                
                # 2D detections
                det_vis = self.visualizer.visualize_detections_2d(
                    image, results['detections_2d'], sample_data['boxes_2d'],
                    save_path=os.path.join(self.output_dir, f"{sample_token}_2d.{img_format}"),
                    show_gt=show_gt_2d,
                    show_pred=show_pred_2d,
                    show_class_labels=show_labels_2d
                )
                
                # BEV visualization
                bev_vis = self.visualizer.plot_bev(
                    gt_objects=results['gt_bev'],
                    pred_objects=results['predictions_bev'],
                    title=f"BEV - Sample {sample_token[:8]}",
                    save_path=os.path.join(self.output_dir, f"{sample_token}_bev.{img_format}"),
                    show_class_labels=show_labels_bev,
                    show_gt=show_gt_bev,
                    show_pred=show_pred_bev
                )
                
                # Comparison plot
                self.visualizer.create_comparison_plot(
                    image, results['detections_2d'],
                    results['gt_bev'], results['predictions_bev'],
                    sample_token, save_dir=self.output_dir,
                    gt_boxes_2d=sample_data['boxes_2d'],
                    show_class_labels_2d=show_labels_2d,
                    show_gt_2d=show_gt_2d,
                    show_pred_2d=show_pred_2d,
                    show_class_labels_bev=show_labels_bev,
                    show_gt_bev=show_gt_bev,
                    show_pred_bev=show_pred_bev
                )
            
            # Collect metrics
            if 'metrics' in results and results['metrics']:
                all_metrics.append(results['metrics'])
        
        # Compute average metrics
        if all_metrics:
            avg_metrics = {
                'position_mse': np.mean([m['position_mse'] for m in all_metrics]),
                'position_mae': np.mean([m['position_mae'] for m in all_metrics]),
                'orientation_error': np.mean([m['orientation_error'] for m in all_metrics]),
                'size_error': np.mean([m['size_error'] for m in all_metrics]),
                'num_samples': len(all_metrics)
            }
            
            print("\n" + "="*80)
            print("Average Metrics:")
            print("-"*80)
            print(f"Position MSE: {avg_metrics['position_mse']:.4f} m²")
            print(f"Position MAE: {avg_metrics['position_mae']:.4f} m")
            print(f"Orientation Error: {avg_metrics['orientation_error']:.4f} rad")
            print(f"Size Error: {avg_metrics['size_error']:.4f} m")
            print(f"Number of Samples: {avg_metrics['num_samples']}")
            print("="*80)
            
            # Save metrics
            import json
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(avg_metrics, f, indent=4)
            print(f"\nMetrics saved to {metrics_path}")


def main():
    """
    Main function to run pipeline.
    """
    parser = argparse.ArgumentParser(description='Run 2D-to-BEV Detection Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--sample-token', type=str, default=None,
                       help='Specific sample token to process')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize NuScenes
    print("Initializing nuScenes dataset...")
    nusc = NuScenes(version=config['data']['version'],
                    dataroot=config['data']['nuscenes_root'],
                    verbose=True)
    
    # Initialize pipeline
    pipeline = BEVPipeline(config, checkpoint_path=args.checkpoint)
    
    # Run pipeline
    if args.sample_token:
        samples = [args.sample_token]
    else:
        # Use test split
        all_samples = [s['token'] for s in nusc.sample]
        split_idx = int(len(all_samples) * config['data']['train_split_ratio'])
        samples = all_samples[split_idx:]
    
    pipeline.run_on_dataset(nusc, samples, max_samples=args.max_samples)
    
    print("\nPipeline completed successfully!")


if __name__ == '__main__':
    main()
