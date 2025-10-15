#!/bin/bash

# ================================================================================
# Simple Tmux Training Script
# ================================================================================
# Quick script to start training in tmux with auto-restart
# ================================================================================

SESSION_NAME="yolo_training"
TRAINING_COMMAND="python scripts/train.py --config configs/train_config.yaml --data configs/dataset.yaml --resume results/runs/train/weights/best.pt --epochs 200 --batch 12 --device 0"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting YOLO Training in Tmux${NC}"

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Killing existing session: $SESSION_NAME${NC}"
    tmux kill-session -t "$SESSION_NAME"
fi

# Create new tmux session
echo -e "${GREEN}Creating tmux session: $SESSION_NAME${NC}"
tmux new-session -d -s "$SESSION_NAME"

# Set up the session
tmux send-keys -t "$SESSION_NAME" "echo 'Starting YOLO Training...'" Enter
tmux send-keys -t "$SESSION_NAME" "echo 'Session started at: $(date)'" Enter
tmux send-keys -t "$SESSION_NAME" "nvidia-smi" Enter
tmux send-keys -t "$SESSION_NAME" "echo '====================================='" Enter

# Start training with auto-restart loop
tmux send-keys -t "$SESSION_NAME" 'while true; do' Enter
tmux send-keys -t "$SESSION_NAME" "  echo 'Starting training at: \$(date)'" Enter
tmux send-keys -t "$SESSION_NAME" "  $TRAINING_COMMAND" Enter
tmux send-keys -t "$SESSION_NAME" "  EXIT_CODE=\$?" Enter
tmux send-keys -t "$SESSION_NAME" "  echo 'Training ended with exit code: \$EXIT_CODE at \$(date)'" Enter
tmux send-keys -t "$SESSION_NAME" '  if [ $EXIT_CODE -eq 0 ]; then' Enter
tmux send-keys -t "$SESSION_NAME" "    echo 'Training completed successfully! Exiting.'" Enter
tmux send-keys -t "$SESSION_NAME" '    break' Enter
tmux send-keys -t "$SESSION_NAME" '  else' Enter
tmux send-keys -t "$SESSION_NAME" "    echo 'Training failed or crashed. Restarting in 30 seconds...'" Enter
tmux send-keys -t "$SESSION_NAME" "    echo 'Clearing GPU memory...'" Enter
tmux send-keys -t "$SESSION_NAME" "    python -c 'import torch; torch.cuda.empty_cache()'" Enter
tmux send-keys -t "$SESSION_NAME" "    sleep 30" Enter
tmux send-keys -t "$SESSION_NAME" "    echo 'Restarting training...'" Enter
tmux send-keys -t "$SESSION_NAME" '  fi' Enter
tmux send-keys -t "$SESSION_NAME" 'done' Enter

echo -e "${GREEN}Training started in tmux session: $SESSION_NAME${NC}"
echo ""
echo "Useful commands:"
echo "  Attach to session:    tmux attach -t $SESSION_NAME"
echo "  Detach from session:  Ctrl+B, then D"
echo "  Kill session:         tmux kill-session -t $SESSION_NAME"
echo "  List sessions:        tmux list-sessions"
echo ""
echo "To attach now, run: tmux attach -t $SESSION_NAME"
