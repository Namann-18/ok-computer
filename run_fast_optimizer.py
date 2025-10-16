#!/usr/bin/env python3
"""
Fast 2-Hour Optimizer - Maximum impact in minimum time
Target: Push mAP@50 from 85.8% to as close to 90% as possible in 2 hours
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Configuration
    BEST_MODEL = "optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt"
    DATASET_CONFIG = "configs/dataset.yaml"
    TRAIN_CONFIG = "configs/train_config.yaml"
    BASELINE = 0.858
    TARGET = 0.90
    FINAL_DIR = "./final"
    
    # FAST STRATEGIES - 2 hour budget
    # Each runs ~40-50 epochs max (enough to see improvement)
    STRATEGIES = [
        "strategy_fast_boost",      # 60 epochs, aggressive params (45 min)
        "strategy_quick_tune",      # 50 epochs, loss tuning (35 min)
    ]
    
    print("="*80)
    print("⚡ FAST 2-HOUR OPTIMIZER - Maximum Impact Edition")
    print("="*80)
    print(f"⏱️  Time budget: 2 hours")
    print(f"📊 Current baseline: {BASELINE:.4f} (85.8%)")
    print(f"🎯 Target score: {TARGET:.4f} (90%)")
    print(f"💾 Best models saved to: {FINAL_DIR}/")
    print(f"\n🚀 Quick strategies (2 attempts):")
    for i, s in enumerate(STRATEGIES, 1):
        print(f"   {i}. {s}")
    print("\n⏳ Estimated time: ~90-120 minutes total")
    print("="*80)
    
    if not Path(BEST_MODEL).exists():
        print(f"❌ Model not found: {BEST_MODEL}")
        sys.exit(1)
    
    response = input("\n🚀 Start fast optimization? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        sys.exit(0)
    
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
    
    print(f"\n⚡ Starting fast optimization...")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Optimization complete!")
        print(f"📁 Check {FINAL_DIR}/ for improved models")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()
