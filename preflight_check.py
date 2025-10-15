#!/usr/bin/env python3
"""
Pre-flight check script to validate setup before running optimization.
"""

import sys
from pathlib import Path
import subprocess

print("="*80)
print("Pre-Flight Check for YOLOv8 Optimization Pipeline")
print("="*80)
print()

checks_passed = 0
checks_failed = 0
warnings = 0

# Check 1: Python version
print("✓ Checking Python version...", end=" ")
version = sys.version_info
if version.major >= 3 and version.minor >= 8:
    print(f"OK ({version.major}.{version.minor}.{version.micro})")
    checks_passed += 1
else:
    print(f"FAIL (Need Python 3.8+, found {version.major}.{version.minor})")
    checks_failed += 1

# Check 2: Required packages
print("✓ Checking required packages...")
required_packages = [
    'ultralytics',
    'torch',
    'cv2',
    'yaml',
    'numpy',
    'albumentations',
    'tqdm'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
        checks_passed += 1
    except ImportError:
        print(f"  ✗ {package} - MISSING")
        checks_failed += 1

# Check 3: CUDA availability
print("✓ Checking CUDA availability...", end=" ")
try:
    import torch
    if torch.cuda.is_available():
        print(f"OK (Device: {torch.cuda.get_device_name(0)})")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        checks_passed += 1
    else:
        print("WARNING - No CUDA available, will use CPU (slower)")
        warnings += 1
except Exception as e:
    print(f"ERROR - {e}")
    checks_failed += 1

# Check 4: Model file
print("✓ Checking base model...", end=" ")
model_paths = [
    "results/runs/train/weights/best.pt",
    "runs/train/weights/best.pt",
    "best.pt"
]
model_found = False
for model_path in model_paths:
    if Path(model_path).exists():
        print(f"OK ({model_path})")
        checks_passed += 1
        model_found = True
        break

if not model_found:
    print("WARNING - Model not found in common locations")
    print("  Please specify correct path when running scripts")
    warnings += 1

# Check 5: Dataset configuration
print("✓ Checking dataset configuration...", end=" ")
if Path("configs/dataset.yaml").exists():
    print("OK")
    checks_passed += 1
else:
    print("FAIL - configs/dataset.yaml not found")
    checks_failed += 1

# Check 6: Training configuration
print("✓ Checking training configuration...", end=" ")
if Path("configs/train_config.yaml").exists():
    print("OK")
    checks_passed += 1
else:
    print("FAIL - configs/train_config.yaml not found")
    checks_failed += 1

# Check 7: Dataset files
print("✓ Checking dataset structure...", end=" ")
if Path("datasets/train/images").exists():
    train_images = list(Path("datasets/train/images").glob("*.jpg")) + \
                  list(Path("datasets/train/images").glob("*.png"))
    print(f"OK ({len(train_images)} training images)")
    checks_passed += 1
else:
    print("WARNING - Training images not found")
    warnings += 1

# Check 8: Disk space
print("✓ Checking disk space...", end=" ")
try:
    if sys.platform == 'win32':
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        if free_gb > 20:
            print(f"OK ({free_gb:.1f} GB free)")
            checks_passed += 1
        else:
            print(f"WARNING - Low disk space ({free_gb:.1f} GB free)")
            warnings += 1
    else:
        print("SKIP (not Windows)")
except Exception as e:
    print(f"SKIP ({e})")

# Check 9: New scripts
print("✓ Checking optimization scripts...")
scripts = [
    'scripts/incremental_learning.py',
    'scripts/advanced_augmentation.py',
    'scripts/model_optimizer.py',
    'scripts/model_ensemble.py',
    'scripts/master_pipeline.py'
]

for script in scripts:
    if Path(script).exists():
        print(f"  ✓ {script}")
        checks_passed += 1
    else:
        print(f"  ✗ {script} - MISSING")
        checks_failed += 1

print()
print("="*80)
print("Summary:")
print(f"  ✓ Checks Passed: {checks_passed}")
if warnings > 0:
    print(f"  ⚠ Warnings: {warnings}")
if checks_failed > 0:
    print(f"  ✗ Checks Failed: {checks_failed}")
print("="*80)
print()

if checks_failed > 0:
    print("❌ Some checks failed. Please fix issues before running optimization.")
    print()
    print("To install missing packages:")
    print("  pip install ultralytics albumentations optuna torch torchvision")
    sys.exit(1)
elif warnings > 0:
    print("⚠️  Ready with warnings. You can proceed but may encounter issues.")
    print()
    print("Recommended actions:")
    if not model_found:
        print("  - Specify correct model path when running scripts")
    print("  - Ensure sufficient disk space (20+ GB recommended)")
    print()
    sys.exit(0)
else:
    print("✅ All checks passed! Ready to optimize.")
    print()
    print("Quick start:")
    print("  python scripts/master_pipeline.py --model results/runs/train/weights/best.pt")
    print()
    print("Or use interactive script:")
    print("  .\\quick_start.ps1  (Windows PowerShell)")
    print()
    sys.exit(0)
