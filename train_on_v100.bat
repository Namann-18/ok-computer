@echo off
REM ============================================================================
REM Domain-Adapted Training Script for Google Cloud V100 (Windows)
REM ============================================================================
REM Optimized configuration for NVIDIA V100 16GB GPU
REM Expected training time: 12-16 hours (350 epochs, batch size 32)
REM ============================================================================

echo ==============================================================================
echo Starting Domain-Adapted Training on V100
echo ==============================================================================

REM Check GPU
echo.
echo Checking GPU...
nvidia-smi

REM Verify CUDA availability
echo.
echo Checking PyTorch CUDA...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

REM Create necessary directories
if not exist logs mkdir logs
if not exist models mkdir models
if not exist results\runs mkdir results\runs

REM Start training
echo.
echo ==============================================================================
echo Starting Training...
echo Configuration:
echo   - Model: YOLOv8m
echo   - Epochs: 350
echo   - Batch Size: 32
echo   - Image Size: 640
echo   - GPU: V100 16GB
echo   - Cache: Enabled
echo   - Workers: 8
echo   - Expected Time: 12-16 hours
echo ==============================================================================
echo.

REM Run training
python scripts/train.py --config configs/train_config.yaml --data configs/dataset.yaml --log-dir logs

echo.
echo ==============================================================================
echo Training completed!
echo ==============================================================================
echo.
echo Check results:
echo   - Best model: models/best.pt
echo   - Training logs: logs/
echo   - TensorBoard: tensorboard --logdir results/runs
echo.
echo Next steps:
echo   1. Test on real images: python scripts/inference_tta.py --model models/best.pt --source path/to/real/images
echo   2. Update API model path in api.py
echo   3. Deploy updated model
echo ==============================================================================

pause
