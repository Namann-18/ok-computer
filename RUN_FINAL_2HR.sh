#!/bin/bash
# ============================================================================
# FINAL 2-HOUR OPTIMIZATION RUN - COPY THIS COMMAND
# ============================================================================
# Current Baseline: 86.2% mAP@50 (from epoch 40)
# Target: 90% mAP@50 (3.8% improvement needed)
# Time Budget: 2 hours maximum
# ============================================================================

cd ~/safeorbit/ok-computer && python scripts/model_optimizer.py --model results/improved_model/train/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.862 --final-dir ./final --strategies strategy_fast_boost strategy_quick_tune
