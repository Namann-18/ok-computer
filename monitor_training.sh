#!/bin/bash
# ==============================================================================
# Quick Monitoring Script for V100 Training
# ==============================================================================
# Run this to get a quick status update on your training
# Usage: ./monitor_training.sh
# ==============================================================================

echo "=============================================================================="
echo "Training Status Check"
echo "=============================================================================="
echo ""

# Check if training process is running
echo "1. Training Process Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ps aux | grep -q "[p]ython scripts/train.py"; then
    echo "✅ Training is RUNNING"
    ps aux | grep "[p]ython scripts/train.py" | awk '{print "   PID: " $2 " | CPU: " $3 "% | Memory: " $4 "% | Started: " $9}'
else
    echo "❌ Training is NOT running"
fi
echo ""

# Check GPU utilization
echo "2. GPU Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
awk -F', ' '{printf "   GPU: %s\n   Utilization: %s%%\n   Memory: %s MB / %s MB (%.1f%%)\n   Temperature: %s°C\n   Power: %s W\n", $1, $2, $3, $4, ($3/$4)*100, $5, $6}'
echo ""

# Check training progress from log
echo "3. Training Progress:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "training_v100.log" ]; then
    # Get last epoch info
    LAST_EPOCH=$(grep -oP "Epoch \K\d+/\d+" training_v100.log | tail -1)
    
    if [ ! -z "$LAST_EPOCH" ]; then
        echo "   Last Epoch: $LAST_EPOCH"
        
        # Get latest metrics
        LATEST_MAP=$(grep "mAP50" training_v100.log | tail -1 | grep -oP "mAP50.*" || echo "Calculating...")
        LATEST_LOSS=$(grep "loss" training_v100.log | tail -3 | head -1 || echo "Calculating...")
        
        echo "   Latest mAP: $LATEST_MAP"
        echo "   Latest Loss: $LATEST_LOSS"
        
        # Estimate time remaining
        CURRENT_EPOCH=$(echo $LAST_EPOCH | cut -d'/' -f1)
        TOTAL_EPOCHS=$(echo $LAST_EPOCH | cut -d'/' -f2)
        REMAINING=$((TOTAL_EPOCHS - CURRENT_EPOCH))
        TIME_REMAINING=$((REMAINING * 2)) # Assuming 2 min per epoch
        HOURS=$((TIME_REMAINING / 60))
        MINUTES=$((TIME_REMAINING % 60))
        
        echo "   Remaining: ~$HOURS hours $MINUTES minutes"
    else
        echo "   Training just started, no metrics yet..."
    fi
else
    echo "   ⚠️  Log file not found: training_v100.log"
fi
echo ""

# Check disk space
echo "4. Disk Space:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
df -h . | tail -1 | awk '{print "   Available: " $4 " / " $2 " (" $5 " used)"}'
echo ""

# Check latest checkpoint
echo "5. Checkpoints:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "results/runs/train/weights" ]; then
    echo "   Saved checkpoints:"
    ls -lh results/runs/train/weights/*.pt 2>/dev/null | awk '{print "   - " $9 " (" $5 ")"}'
elif [ -d "models" ]; then
    echo "   Saved models:"
    ls -lh models/*.pt 2>/dev/null | awk '{print "   - " $9 " (" $5 ")"}'
else
    echo "   No checkpoints yet (training in progress)"
fi
echo ""

echo "=============================================================================="
echo "Quick Commands:"
echo "=============================================================================="
echo "  View live log:      tail -f training_v100.log"
echo "  Watch GPU:          watch -n 1 nvidia-smi"
echo "  TensorBoard:        tensorboard --logdir results/runs --host 0.0.0.0"
echo "  Stop training:      kill $(pgrep -f 'python scripts/train.py')"
echo "=============================================================================="
echo ""
