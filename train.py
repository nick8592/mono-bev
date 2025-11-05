"""
Training Script for 2D-to-BEV Regression Model
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import json

from data_loader import create_dataloaders
from detector import YOLODetector
from bev_transform import BEVTransformModel, BEVTransformLoss, prepare_batch_for_bev_model, prepare_bev_targets
from visualizer import BEVVisualizer, compute_bev_metrics


class BEVTrainer:
    """
    Trainer class for BEV transformation model.
    """
    
    def __init__(self, config: dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['detector']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.log_dir = config['logging']['log_dir']
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize models
        print("Initializing models...")
        self.detector = YOLODetector(config)
        self.bev_model = BEVTransformModel(config).to(self.device)
        self.criterion = BEVTransformLoss(config)
        
        # Initialize optimizer
        if config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.bev_model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        elif config['training']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.bev_model.parameters(),
                lr=config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=config['training']['weight_decay']
            )
        
        # Initialize scheduler
        if config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['training']['step_size'],
                gamma=config['training']['gamma']
            )
        elif config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['num_epochs']
            )
        else:
            self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Visualizer
        self.visualizer = BEVVisualizer(config)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.bev_model.train()
        total_loss = 0.0
        num_batches = 0
        num_valid_samples = 0
        num_skipped_no_detections = 0
        num_skipped_no_targets = 0
        
        # Gradient accumulation to simulate larger batch size
        accumulation_steps = 8  # Simulates batch_size * 8
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            image_paths = batch['image_paths']
            gt_boxes_2d = batch['boxes_2d']
            bev_coords = batch['bev_coords']
            camera_intrinsics = batch['camera_intrinsics']
            
            # Convert images from tensor to numpy for YOLO (expects HWC format, 0-255 range)
            images_np = []
            for i in range(images.shape[0]):
                # Get single image: (C, H, W) -> (H, W, C)
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                # Scale to 0-255 and convert to uint8
                img = (img * 255).clip(0, 255).astype(np.uint8)
                images_np.append(img)
            
            # Run YOLO detection
            detections = self.detector.detect_batch(images_np)
            
            # Debug: Count detections
            total_detections = sum(len(d) for d in detections)
            if total_detections == 0:
                num_skipped_no_detections += 1
                continue
            
            # Prepare batch for BEV model
            image_crops, bbox_2d, cam_intrinsic, class_one_hot = prepare_batch_for_bev_model(
                detections, images, camera_intrinsics,
                target_size=tuple(self.config['bev_model']['input_size']),
                num_classes=self.config['bev_model']['num_classes']
            )
            
            if len(image_crops) == 0:
                num_skipped_no_detections += 1
                continue  # Skip batch if no detections
            
            # Prepare targets with ground truth 2D boxes for matching
            targets, matched_indices = prepare_bev_targets(bev_coords, detections, gt_boxes_2d)
            
            if len(targets['position']) == 0:
                num_skipped_no_targets += 1
                continue  # Skip if no valid targets
            
            num_valid_samples += len(targets['position'])
            
            # Filter predictions to only matched detections
            matched_global_indices = [idx for _, idx in matched_indices]
            image_crops = image_crops[matched_global_indices]
            bbox_2d = bbox_2d[matched_global_indices]
            cam_intrinsic = cam_intrinsic[matched_global_indices]
            class_one_hot = class_one_hot[matched_global_indices]
            
            # Move to device
            image_crops = image_crops.to(self.device)
            bbox_2d = bbox_2d.to(self.device)
            cam_intrinsic = cam_intrinsic.to(self.device)
            class_one_hot = class_one_hot.to(self.device)
            
            for key in targets:
                targets[key] = targets[key].to(self.device)
            
            # Forward pass
            predictions = self.bev_model(image_crops, bbox_2d, cam_intrinsic, class_one_hot)
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps  # Unscale for logging
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'pos': f"{loss_dict['position']:.4f}",
                'orient': f"{loss_dict['orientation']:.4f}"
            })
            
            # Clear GPU cache periodically
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final optimizer step if there are remaining gradients
        if num_batches % accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Print statistics
        print(f"\nTraining Statistics:")
        print(f"  Valid samples: {num_valid_samples}")
        print(f"  Skipped (no detections): {num_skipped_no_detections}")
        print(f"  Skipped (no targets): {num_skipped_no_targets}")
        print(f"  Batches processed: {num_batches}")
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader, epoch: int) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        self.bev_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                gt_boxes_2d = batch['boxes_2d']
                bev_coords = batch['bev_coords']
                camera_intrinsics = batch['camera_intrinsics']
                
                # Convert images from tensor to numpy for YOLO (expects HWC format, 0-255 range)
                images_np = []
                for i in range(images.shape[0]):
                    # Get single image: (C, H, W) -> (H, W, C)
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    # Scale to 0-255 and convert to uint8
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    images_np.append(img)
                
                # Run YOLO detection
                detections = self.detector.detect_batch(images_np)
                
                # Prepare batch for BEV model
                image_crops, bbox_2d, cam_intrinsic, class_one_hot = prepare_batch_for_bev_model(
                    detections, images, camera_intrinsics,
                    target_size=tuple(self.config['bev_model']['input_size']),
                    num_classes=self.config['bev_model']['num_classes']
                )
                
                if len(image_crops) == 0:
                    continue
                
                # Prepare targets with ground truth 2D boxes for matching
                targets, matched_indices = prepare_bev_targets(bev_coords, detections, gt_boxes_2d)
                
                if len(targets['position']) == 0:
                    continue
                
                # Filter predictions to only matched detections
                matched_global_indices = [idx for _, idx in matched_indices]
                image_crops = image_crops[matched_global_indices]
                bbox_2d = bbox_2d[matched_global_indices]
                cam_intrinsic = cam_intrinsic[matched_global_indices]
                class_one_hot = class_one_hot[matched_global_indices]
                
                # Move to device
                image_crops = image_crops.to(self.device)
                bbox_2d = bbox_2d.to(self.device)
                cam_intrinsic = cam_intrinsic.to(self.device)
                class_one_hot = class_one_hot.to(self.device)
                
                for key in targets:
                    targets[key] = targets[key].to(self.device)
                
                # Forward pass
                predictions = self.bev_model(image_crops, bbox_2d, cam_intrinsic, class_one_hot)
                
                # Compute loss
                loss, loss_dict = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80 + "\n")
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                print(f"✓ New best model saved! Val Loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        
        # Save training history
        self.save_training_history()
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.bev_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def save_training_history(self):
        """
        Save training history to JSON.
        """
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        path = os.path.join(self.log_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Training history saved to {path}")


def main():
    """
    Main function to run training.
    """
    parser = argparse.ArgumentParser(description='Train BEV Transformation Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, nusc = create_dataloaders(config)
    
    # Initialize trainer
    trainer = BEVTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
