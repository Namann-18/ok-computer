# ============================================================================
# YOLOv8 Model Optimizer - Linux Cloud Terminal Commands
# Target: Improve mAP@50 from 85.8% to 90%+
# ============================================================================

# 1. Navigate to project directory
cd /path/to/safeorbit/ok-computer

# 2. Activate virtual environment (if using one)
source venv/bin/activate
# OR if using conda:
# conda activate your_env_name

# 3. Install/verify dependencies
pip install ultralytics pyyaml optuna

# ============================================================================
# OPTION A: Quick Run (Recommended - uses preset strategies)
# ============================================================================

# Run the quick optimizer script
python run_optimizer.py

# This will:
# - Use your best model (85.8% mAP@50)
# - Try 4 optimized strategies
# - Save improved models to ./final/ directory
# - Auto-confirm or run with -y flag

# ============================================================================
# OPTION B: Manual Run (Full control)
# ============================================================================

# Run with all new strategies (600 epochs, optimized params)
python scripts/model_optimizer.py \
  --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --goal map50 \
  --target 0.90 \
  --baseline 0.858 \
  --final-dir ./final \
  --strategies strategy_5_ultra_extended strategy_7_loss_tuned strategy_6_advanced_aug strategy_8_mega_batch

# ============================================================================
# OPTION C: Try Single Strategy (Fastest - for testing)
# ============================================================================

# Try just one strategy (ultra extended training)
python scripts/model_optimizer.py \
  --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --goal map50 \
  --target 0.90 \
  --baseline 0.858 \
  --final-dir ./final \
  --strategies strategy_5_ultra_extended

# ============================================================================
# OPTION D: Run in Background with nohup (Recommended for long training)
# ============================================================================

# Run in background and save output to log file
nohup python scripts/model_optimizer.py \
  --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --goal map50 \
  --target 0.90 \
  --baseline 0.858 \
  --final-dir ./final \
  --strategies strategy_5_ultra_extended strategy_7_loss_tuned strategy_6_advanced_aug \
  > optimizer_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get the process ID
echo $!

# Monitor progress
tail -f optimizer_*.log

# ============================================================================
# OPTION E: Run with tmux/screen (Best for cloud - survives disconnection)
# ============================================================================

# Start tmux session
tmux new -s optimizer

# Inside tmux, run optimizer
python scripts/model_optimizer.py \
  --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --goal map50 \
  --target 0.90 \
  --baseline 0.858 \
  --final-dir ./final \
  --strategies strategy_5_ultra_extended strategy_7_loss_tuned strategy_6_advanced_aug

# Detach from tmux: Press Ctrl+B then D
# Reattach later: tmux attach -t optimizer
# Kill session: tmux kill-session -t optimizer

# ============================================================================
# Available Strategies (pick what to run)
# ============================================================================

# strategy_1_focal_loss          - Enhanced focal loss (moderate improvement)
# strategy_2_aggressive_aug      - Strong augmentation (good for robustness)
# strategy_3_optimized_lr        - Fine-tuned learning rate (stable improvement)
# strategy_4_extended_training   - 500 epochs (already tried: 85.8%)
# strategy_5_ultra_extended      - 600 epochs + optimized (BEST CHANCE for 90%)
# strategy_6_advanced_aug        - Advanced aug + 500 epochs (high potential)
# strategy_7_loss_tuned          - Fine-tuned losses (balanced improvement)
# strategy_8_mega_batch          - Large batch training (if GPU memory allows)

# ============================================================================
# Monitoring Commands
# ============================================================================

# Check GPU usage
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check running processes
ps aux | grep model_optimizer

# Monitor log files
tail -f logs/*.log

# Check final directory for new models
ls -lht final/

# View latest metrics
cat final/metrics_*.json | tail -50

# ============================================================================
# After Training: Check Results
# ============================================================================

# List saved models in final directory
ls -lh final/

# View best model metrics
cat final/metrics_*.json | python -m json.tool

# Check optimization results
cat optimization_runs/optimization_results.json | python -m json.tool

# Copy best model to production location (if target reached)
cp final/best.pt ../app/assets/models/

# ============================================================================
# Troubleshooting
# ============================================================================

# If CUDA out of memory, reduce batch size:
# Edit configs/train_config.yaml:
# training:
#   batch_size: 8  # Reduce from 16

# If process killed, check system memory:
free -h

# Monitor disk space:
df -h

# Clear old optimization runs to free space:
rm -rf optimization_runs/strategy_1_*
rm -rf optimization_runs/strategy_2_*
rm -rf optimization_runs/strategy_3_*

# ============================================================================
# Expected Timeline
# ============================================================================

# strategy_5_ultra_extended:    ~15-20 hours (600 epochs)
# strategy_7_loss_tuned:        ~12-15 hours (500 epochs)
# strategy_6_advanced_aug:      ~12-15 hours (500 epochs)
# strategy_8_mega_batch:        ~10-12 hours (550 epochs)

# Total (all 4 strategies):     ~50-60 hours
# With early stopping:          ~30-40 hours (if target reached)

# ============================================================================
# Quick Start (Copy-Paste Ready)
# ============================================================================

cd /path/to/safeorbit/ok-computer && \
source venv/bin/activate && \
tmux new -s optimizer

# Then inside tmux:
python scripts/model_optimizer.py \
  --model optimization_runs/strategy_4_extended_training_20251015_101328/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --target 0.90 \
  --baseline 0.858 \
  --final-dir ./final \
  --strategies strategy_5_ultra_extended strategy_7_loss_tuned

# Detach: Ctrl+B then D
# Monitor: tmux attach -t optimizer
