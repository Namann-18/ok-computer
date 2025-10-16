#!/bin/bash
# ============================================================================
# FINAL 90% TARGET RUN - 2 HOUR OPTIMIZATION
# ============================================================================
# Current Status: 86.6% mAP@50 (from improved_model)
# Target: 90.0% mAP@50 
# Improvement Needed: 3.4% (very achievable!)
# Time Budget: 2 hours maximum
# Strategy: 2 ultra-fast aggressive strategies optimized for rapid improvement
# ============================================================================

cd ~/safeorbit/ok-computer

# RECOMMENDED: Run both fast strategies in sequence (~2 hours total)
tmux new -s optimizer_90

python scripts/model_optimizer.py \
  --model results/improved_model/train/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --target 0.90 \
  --baseline 0.866 \
  --final-dir ./final \
  --strategies strategy_fast_boost strategy_quick_tune

# To detach from tmux: Ctrl+B then D
# To reattach: tmux attach -t optimizer_90
# To monitor: watch -n 1 nvidia-smi
