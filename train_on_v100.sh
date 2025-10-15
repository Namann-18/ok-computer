#!/bin/bash
################################################################################
# Domain-Adapted Training Script for Google Cloud V100
################################################################################
# Optimized configuration for NVIDIA V100 16GB GPU
# Expected training time: 12-16 hours (350 epochs, batch size 32)
################################################################################

echo "=============================================================================="
echo "Starting Domain-Adapted Training on V100"
echo "=============================================================================="

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi

# Verify CUDA availability
echo ""
echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p results/runs

# Start training with optimal V100 settings
echo ""
echo "=============================================================================="
echo "Starting Training..."
echo "Configuration:"
echo "  - Model: YOLOv8m"
echo "  - Epochs: 350"
echo "  - Batch Size: 32"
echo "  - Image Size: 640"
echo "  - GPU: V100 16GB"
echo "  - Cache: Enabled"
echo "  - Workers: 8"
echo "  - Expected Time: 12-16 hours"
echo "=============================================================================="
echo ""

# Run training with nohup to survive disconnection
nohup python scripts/train.py \
    --config configs/train_config.yaml \
    --data configs/dataset.yaml \
    --log-dir logs \
    > training_v100.log 2>&1 &

# Get process ID
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo ""
echo "Monitor progress:"
echo "  - tail -f training_v100.log"
echo "  - tensorboard --logdir results/runs"
echo "  - nvidia-smi"
echo ""
echo "Training is running in background. Safe to disconnect."
echo "=============================================================================="
