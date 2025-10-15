"""
================================================================================
Quick Setup Script for Domain-Adapted Training
================================================================================
Prepares environment and validates setup before training.

Usage:
    python setup_domain_adaptation.py
    
Author: Space Station Safety Detection Team
Date: 2025-10-16
================================================================================
"""

import sys
from pathlib import Path
import subprocess
import importlib.util

def check_import(package_name: str, install_name: str = None) -> bool:
    """Check if a package is installed."""
    install_name = install_name or package_name
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ {package_name} not found")
        return False
    print(f"✅ {package_name} installed")
    return True

def install_package(package_name: str):
    """Install a package using pip."""
    print(f"\nInstalling {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def main():
    print("="*80)
    print("Domain Adaptation Setup")
    print("="*80)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required (current: {sys.version_info.major}.{sys.version_info.minor})")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check required packages
    print("\n2. Checking dependencies...")
    required_packages = {
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'yaml': 'pyyaml',
        'albumentations': 'albumentations',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
    }
    
    missing_packages = []
    for import_name, install_name in required_packages.items():
        if not check_import(import_name):
            missing_packages.append(install_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\n3. Installing missing packages: {', '.join(missing_packages)}")
        install_choice = input("Install missing packages? (y/n): ").lower()
        if install_choice == 'y':
            for package in missing_packages:
                if not install_package(package):
                    print(f"\n⚠ Warning: {package} installation failed. Please install manually.")
        else:
            print("\n⚠ Please install missing packages manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    else:
        print("\n✅ All dependencies installed")
    
    # Check project structure
    print("\n4. Checking project structure...")
    required_dirs = [
        'configs',
        'datasets',
        'scripts',
        'utils',
    ]
    
    required_files = [
        'configs/train_config.yaml',
        'configs/dataset.yaml',
        'scripts/train.py',
        'scripts/domain_adaptation.py',
        'scripts/inference_tta.py',
    ]
    
    project_ok = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")
            project_ok = False
    
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            project_ok = False
    
    if not project_ok:
        print("\n❌ Project structure incomplete")
        return False
    
    # Check GPU availability
    print("\n5. Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 12:
                print(f"⚠ Warning: Low GPU memory. Recommend reducing batch size to 16")
        else:
            print("⚠ No GPU detected. Training will be very slow on CPU.")
            cpu_choice = input("Continue with CPU training? (y/n): ").lower()
            if cpu_choice != 'y':
                return False
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
    
    # Check dataset
    print("\n6. Checking dataset...")
    dataset_path = Path('datasets')
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = dataset_path / split / 'images'
        label_dir = dataset_path / split / 'labels'
        
        if img_dir.exists() and label_dir.exists():
            img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
            label_count = len(list(label_dir.glob('*.txt')))
            print(f"✅ {split}: {img_count} images, {label_count} labels")
            
            if img_count != label_count:
                print(f"⚠ Warning: Image/label count mismatch in {split}")
        else:
            print(f"❌ {split} split not found")
            project_ok = False
    
    if not project_ok:
        print("\n❌ Dataset incomplete")
        return False
    
    # Summary
    print("\n" + "="*80)
    print("Setup Summary")
    print("="*80)
    print("✅ Environment ready for domain-adapted training")
    print("\nNext steps:")
    print("  1. Review DOMAIN_ADAPTATION_GUIDE.md")
    print("  2. Start training:")
    print("     python scripts/train.py --config configs/train_config.yaml")
    print("  3. Monitor progress in logs/ and results/")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
