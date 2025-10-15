#!/bin/bash

# ================================================================================
# YOLO Training Script with Auto-Restart on GPU Crash
# ================================================================================
# This script starts training in a tmux session with automatic restart capability
# for GPU crashes and memory issues on Google Cloud V100
# ================================================================================

# Configuration
SESSION_NAME="yolo_training"
TRAINING_COMMAND="python scripts/train.py --config configs/train_config.yaml --data configs/dataset.yaml --resume results/runs/train/weights/best.pt --epochs 200 --batch 12 --device 0"
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
MAX_RESTARTS=10
RESTART_DELAY=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting YOLO Training with Auto-Restart${NC}"
echo "Session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Max restarts: $MAX_RESTARTS"
echo "=================================="

# Function to check if tmux session exists
session_exists() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

# Function to create tmux session
create_session() {
    echo -e "${GREEN}Creating new tmux session: $SESSION_NAME${NC}"
    tmux new-session -d -s "$SESSION_NAME"
    
    # Set up the session with monitoring
    tmux send-keys -t "$SESSION_NAME" "echo 'Starting YOLO Training with Auto-Restart...'" Enter
    tmux send-keys -t "$SESSION_NAME" "echo 'Session started at: $(date)'" Enter
    tmux send-keys -t "$SESSION_NAME" "echo 'GPU Info:'" Enter
    tmux send-keys -t "$SESSION_NAME" "nvidia-smi" Enter
    tmux send-keys -t "$SESSION_NAME" "echo '====================================='" Enter
}

# Function to start training
start_training() {
    local restart_count=$1
    echo -e "${YELLOW}Starting training (attempt $((restart_count + 1))/$((MAX_RESTARTS + 1)))${NC}"
    
    tmux send-keys -t "$SESSION_NAME" "echo 'Training attempt $((restart_count + 1)) started at: $(date)'" Enter
    tmux send-keys -t "$SESSION_NAME" "echo 'Command: $TRAINING_COMMAND'" Enter
    tmux send-keys -t "$SESSION_NAME" "echo '====================================='" Enter
    
    # Start training with logging
    tmux send-keys -t "$SESSION_NAME" "$TRAINING_COMMAND 2>&1 | tee -a $LOG_FILE" Enter
}

# Function to monitor training and detect crashes
monitor_training() {
    local restart_count=0
    
    while [ $restart_count -le $MAX_RESTARTS ]; do
        echo -e "${GREEN}Monitoring training session...${NC}"
        
        # Wait for training to start
        sleep 10
        
        # Monitor for GPU crashes or training completion
        while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
            # Check if training process is still running
            if ! tmux list-panes -t "$SESSION_NAME" -F "#{pane_current_command}" | grep -q "python"; then
                echo -e "${YELLOW}Training process ended. Checking for completion or crash...${NC}"
                break
            fi
            
            # Check GPU memory usage
            gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
            if [ "$gpu_memory" -gt 15000 ]; then
                echo -e "${RED}Warning: GPU memory usage high ($gpu_memory MB)${NC}"
            fi
            
            sleep 30
        done
        
        # Check if session still exists
        if ! session_exists; then
            echo -e "${RED}Session ended unexpectedly. Creating new session...${NC}"
            create_session
        fi
        
        # Check if training completed successfully
        if grep -q "Training completed successfully" "$LOG_FILE" 2>/dev/null; then
            echo -e "${GREEN}Training completed successfully!${NC}"
            break
        fi
        
        # Check for common crash indicators
        if grep -q -E "(CUDA out of memory|GPU crash|RuntimeError|Killed|Segmentation fault)" "$LOG_FILE" 2>/dev/null; then
            echo -e "${RED}Detected GPU crash or memory issue. Preparing restart...${NC}"
            restart_count=$((restart_count + 1))
            
            if [ $restart_count -le $MAX_RESTARTS ]; then
                echo -e "${YELLOW}Restarting in $RESTART_DELAY seconds... (attempt $((restart_count + 1))/$((MAX_RESTARTS + 1)))${NC}"
                
                # Clear GPU memory
                tmux send-keys -t "$SESSION_NAME" "echo 'Clearing GPU memory...'" Enter
                tmux send-keys -t "$SESSION_NAME" "python -c 'import torch; torch.cuda.empty_cache()'" Enter
                
                # Wait before restart
                sleep $RESTART_DELAY
                
                # Start training again
                start_training $restart_count
            else
                echo -e "${RED}Maximum restart attempts reached. Stopping.${NC}"
                break
            fi
        else
            echo -e "${GREEN}Training appears to have completed normally.${NC}"
            break
        fi
    done
}

# Main execution
main() {
    # Kill existing session if it exists
    if session_exists; then
        echo -e "${YELLOW}Killing existing session: $SESSION_NAME${NC}"
        tmux kill-session -t "$SESSION_NAME"
    fi
    
    # Create new session
    create_session
    
    # Start initial training
    start_training 0
    
    # Monitor and restart if needed
    monitor_training
    
    echo -e "${GREEN}Training session finished.${NC}"
    echo "To attach to the session: tmux attach -t $SESSION_NAME"
    echo "To view logs: tail -f $LOG_FILE"
}

# Run main function
main "$@"
