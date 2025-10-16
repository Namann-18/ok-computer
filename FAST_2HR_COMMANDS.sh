# ============================================================================
# FAST 2-HOUR OPTIMIZER - Copy-Paste Commands
# ============================================================================
# Goal: Maximum improvement from 85.8% → 90% in just 2 hours
# Strategy: Run 2 aggressive short training sessions (60 + 50 epochs)
# ============================================================================

# ==========================
# OPTION 1: RECOMMENDED
# ==========================
# Best 2 fast strategies (~90-120 min total)

cd ~/safeorbit/ok-computer

# Run in foreground (watch progress)
python scripts/model_optimizer.py \
  --model results/improved_model/train/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --target 0.90 \
  --baseline 0.862 \
  --final-dir ./final \
  --strategies strategy_fast_boost strategy_quick_tune


# ==========================
# OPTION 2: SINGLE BEST SHOT
# ==========================
# One ultra-aggressive strategy (~45 min)

# STEP 3C: Run SINGLE strategy (Rapid Convergence - 45 epochs, ~30 min)
python3 scripts/model_optimizer.py \
  --model results/improved_model/train/weights/best.pt \
  --dataset configs/dataset.yaml \
  --train-config configs/train_config.yaml \
  --strategy strategy_rapid_convergence \
  --baseline-score 0.862 \
  --target-score 0.90 \
  --final-dir ./final
# ==========================
# OPTION 3: ALL 3 FAST STRATEGIES
# ==========================
# Try all 3 quick strategies (~2 hours exactly)

python scripts/model_optimizer.py \
  --model results/improved_model/train/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --target 0.90 \
  --baseline 0.862 \
  --final-dir ./final \
  --strategies strategy_fast_boost strategy_quick_tune strategy_rapid_convergence


# ==========================
# WITH TMUX (safer - survives disconnection)
# ==========================

tmux new -s fast_opt

# Inside tmux:
cd ~/safeorbit/ok-computer && python scripts/model_optimizer.py --model results/improved_model/train/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.862 --final-dir ./final --strategies strategy_fast_boost strategy_quick_tune

# Detach: Ctrl+B then D
# Reattach: tmux attach -t fast_opt


# ==========================
# MONITOR PROGRESS
# ==========================

# Watch GPU usage
watch -n 1 nvidia-smi

# Check what's saved
ls -lh final/

# View latest metrics
cat final/metrics_*.json | tail -20


# ============================================================================
# WHAT THESE STRATEGIES DO:
# ============================================================================

# strategy_fast_boost (60 epochs, ~45 min)
# - Very high learning rate (0.0015)
# - Aggressive loss weights (box=9, cls=1.5, dfl=2.5)
# - Strong augmentation (mixup=0.5, copy_paste=0.5)
# - Fast convergence with high momentum (0.97)

# strategy_quick_tune (50 epochs, ~35 min)
# - Maximum loss tuning (box=10, cls=2.0, dfl=3.0)
# - Optimized for precision
# - Fast learning rate decay

# strategy_rapid_convergence (45 epochs, ~30 min)
# - Highest learning rate (0.002)
# - Very high momentum (0.99)
# - Rapid weight updates


# ============================================================================
# EXPECTED RESULTS:
# ============================================================================

# Realistic in 2 hours:
# - 86.2% → 87.5-89% (good improvement, starting from better baseline)
# - 90% is within reach with 3.8% improvement needed
# - Fast strategies optimized for rapid convergence

# Any model > 86.2% is automatically saved to ./final/


# ============================================================================
# ONE-LINER (COPY THIS):
# ============================================================================

cd ~/safeorbit/ok-computer && python scripts/model_optimizer.py --model results/improved_model/train/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.862 --final-dir ./final --strategies strategy_fast_boost strategy_quick_tune
