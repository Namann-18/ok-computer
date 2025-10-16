#!/usr/bin/env python3
"""
Quick run script for model optimization to reach 90% mAP@50
Automatically uses the best existing model as baseline
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Configuration
    BEST_MODEL = "optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt"
    DATASET_CONFIG = "configs/dataset.yaml"
    TRAIN_CONFIG = "configs/train_config.yaml"
    BASELINE = 0.858  # Current best mAP@50
    TARGET = 0.90     # Target 90% mAP@50
    FINAL_DIR = "./final"
    
    # Strategies to try (ordered by likelihood of success)
    STRATEGIES = [
        "strategy_5_ultra_extended",    # 600 epochs with optimized params
        "strategy_7_loss_tuned",        # Fine-tuned loss functions
        "strategy_6_advanced_aug",      # Advanced augmentation
        "strategy_8_mega_batch",        # Large batch training
    ]
    
    print("="*80)
    print("🚀 YOLOv8 Model Optimizer - Target: 90% mAP@50")
    print("="*80)
    print(f"📊 Current baseline: {BASELINE:.4f} (85.8%)")
    print(f"🎯 Target score: {TARGET:.4f} (90%)")
    print(f"📈 Improvement needed: {(TARGET - BASELINE):.4f}")
    print(f"💾 Best models will be saved to: {FINAL_DIR}/")
    print(f"\n🔧 Strategies to try: {len(STRATEGIES)}")
    for i, s in enumerate(STRATEGIES, 1):
        print(f"   {i}. {s}")
    print("="*80)
    
    # Check if model exists
    if not Path(BEST_MODEL).exists():
        print(f"❌ Error: Model not found at {BEST_MODEL}")
        print("   Please train a model first or update the path.")
        sys.exit(1)
    
    # Confirm before starting
    response = input("\n⚠️  This will take several hours. Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/model_optimizer.py",
        "--model", BEST_MODEL,
        "--data", DATASET_CONFIG,
        "--config", TRAIN_CONFIG,
        "--goal", "map50",
        "--target", str(TARGET),
        "--baseline", str(BASELINE),
        "--final-dir", FINAL_DIR,
        "--strategies"
    ] + STRATEGIES
    
    print(f"\n🏃 Starting optimization...")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run optimization
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Optimization complete!")
        print(f"📁 Check {FINAL_DIR}/ directory for improved models")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Optimization failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()
