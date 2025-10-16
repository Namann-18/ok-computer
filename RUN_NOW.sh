# ============================================================================
# QUICK START - Copy and paste this into your Linux cloud terminal
# ============================================================================

# 1. Navigate and activate environment (adjust path as needed)
cd ~/safeorbit/ok-computer && source venv/bin/activate

# 2. Start tmux session (survives disconnection)
tmux new -s optimizer

# 3. Run optimizer (inside tmux) - CHOOSE ONE:

# === RECOMMENDED: Best 2 strategies (fastest path to 90%) ===
python scripts/model_optimizer.py --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.858 --final-dir ./final --strategies strategy_5_ultra_extended strategy_7_loss_tuned

# === FULL RUN: All 4 new strategies (best chances) ===
python scripts/model_optimizer.py --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.858 --final-dir ./final --strategies strategy_5_ultra_extended strategy_7_loss_tuned strategy_6_advanced_aug strategy_8_mega_batch

# === SINGLE TEST: Just one strategy (quick test) ===
python scripts/model_optimizer.py --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.858 --final-dir ./final --strategies strategy_5_ultra_extended

# 4. Detach from tmux (keeps running in background)
#    Press: Ctrl+B, then press: D

# 5. Reattach later to check progress
tmux attach -t optimizer

# 6. Check results
ls -lh final/           # See saved models
cat final/metrics_*.json  # View metrics

# ============================================================================
# EVEN SIMPLER: One-liner with nohup (no tmux needed)
# ============================================================================

cd ~/safeorbit/ok-computer && nohup python scripts/model_optimizer.py --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.858 --final-dir ./final --strategies strategy_5_ultra_extended strategy_7_loss_tuned > optimizer.log 2>&1 & tail -f optimizer.log
