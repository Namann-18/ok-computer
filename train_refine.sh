#!/bin/bash

SESSION_NAME="yolo_refine"

# Start tmux session if not already running
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME
fi

# Send the looped training command into tmux
tmux send-keys -t $SESSION_NAME "
while true; do
    echo '🚀 Starting YOLOv8 refinement training...'
    yolo detect train \
        model=/home/namannayak_16/ok-computer/results/runs/train/weights/best.pt \
        data=/home/namannayak_16/ok-computer/configs/dataset.yaml \
        epochs=200 \
        imgsz=768 \
        lr0=0.0003 \
        cos_lr=True \
        augment=True
    echo '❌ Training crashed or finished. Restarting in 10 seconds...'
    sleep 10
done
" C-m

echo "✅ Training started inside tmux session: $SESSION_NAME"
echo "👉 Attach anytime with: tmux attach -t $SESSION_NAME"
