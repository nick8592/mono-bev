#!/usr/bin/env python3
"""
Quick test to validate image loading and YOLO detection pipeline
"""

import yaml
import torch
import numpy as np
from data_loader import create_dataloaders
from detector import YOLODetector

def test_pipeline():
    print("Testing image loading and detection pipeline...")
    print("-" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, test_loader, nusc = create_dataloaders(config)
    
    # Get one batch
    print("Loading one batch...")
    batch = next(iter(train_loader))
    
    # Check batch structure
    print(f"\nBatch contents:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Number of images: {len(batch['image_paths'])}")
    print(f"  Image dtype: {batch['images'].dtype}")
    print(f"  Image range: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
    
    # Test image conversion
    print("\nTesting image conversion for YOLO...")
    images = batch['images']
    images_np = []
    for i in range(images.shape[0]):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        print(f"  Image {i}: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
        images_np.append(img)
    
    # Test YOLO detector
    print("\nInitializing YOLO detector...")
    detector = YOLODetector(config)
    
    print("\nRunning detection on batch...")
    try:
        detections = detector.detect_batch(images_np)
        print(f"✓ Detection successful!")
        print(f"  Number of images processed: {len(detections)}")
        for i, dets in enumerate(detections):
            print(f"  Image {i}: {len(dets)} objects detected")
            if dets:
                for det in dets[:3]:  # Show first 3 detections
                    print(f"    - {det['class']} (conf: {det['confidence']:.2f})")
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ Pipeline test passed!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    import sys
    success = test_pipeline()
    sys.exit(0 if success else 1)
