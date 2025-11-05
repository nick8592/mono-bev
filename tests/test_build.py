#!/usr/bin/env python3
"""
Quick verification script to test the build and imports
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    print("-" * 60)
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("ultralytics", "Ultralytics (YOLOv11)"),
        ("cv2", "OpenCV"),
        ("nuscenes", "nuScenes DevKit"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
        ("pyquaternion", "PyQuaternion")
    ]
    
    failed = []
    
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name:30s} OK")
        except ImportError as e:
            print(f"✗ {package_name:30s} FAILED: {e}")
            failed.append(package_name)
    
    print("-" * 60)
    
    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        assert False, f"Some packages failed to import: {failed}"
    else:
        print("\n✅ All packages imported successfully!")
        assert True


def test_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    print("-" * 60)
    
    modules = [
        "src.data.data_loader",
        "src.models.detector",
        "src.models.bev_transform",
        "src.visualization.visualizer"
    ]
    
    failed = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:30s} OK")
        except Exception as e:
            print(f"✗ {module_name:30s} FAILED: {e}")
            failed.append(module_name)
    
    print("-" * 60)
    
    if failed:
        print(f"\n❌ {len(failed)} module(s) failed to import:")
        for mod in failed:
            print(f"   - {mod}")
        assert False, f"Some project modules failed to import: {failed}"
    else:
        print("\n✅ All project modules imported successfully!")
        assert True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA availability...")
    print("-" * 60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA is not available (CPU only)")
            print("  Training will be slower without GPU acceleration")
        print("-" * 60)
        assert True
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        print("-" * 60)
        assert False, f"Error checking CUDA: {e}"


def test_config():
    """Test configuration file."""
    print("\nTesting configuration...")
    print("-" * 60)
    
    try:
        import yaml
        import os
        
        config_path = os.path.join('configs', 'default.yaml')
        if not os.path.exists(config_path):
            print(f"✗ {config_path} not found")
            assert False, f"{config_path} not found"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ {config_path} loaded successfully")
        # Check key sections
        required_sections = ['data', 'detector', 'bev_model', 'training', 'visualization']
        for section in required_sections:
            if section in config:
                print(f"  ✓ Section '{section}' found")
            else:
                print(f"  ✗ Section '{section}' missing")
                assert False, f"Section '{section}' missing in config.yaml"
        # Check nuScenes path
        nuscenes_root = config.get('data', {}).get('nuscenes_root', '')
        if os.path.exists(nuscenes_root):
            print(f"  ✓ nuScenes path exists: {nuscenes_root}")
        else:
            print(f"  ⚠ nuScenes path not found: {nuscenes_root}")
            print("    Update config.yaml with correct path to continue")
            # Not a hard assert, just a warning
        print("-" * 60)
        assert True
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        print("-" * 60)
        assert False, f"Error loading config: {e}"


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MONO-BEV BUILD VERIFICATION")
    print("=" * 60 + "\n")
    
    # Remove summary logic for pytest
    pass


if __name__ == '__main__':
    main()
