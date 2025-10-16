@echo off
REM ============================================================================
REM WINDOWS COMMAND - Copy this to run on Windows machine
REM ============================================================================
REM This assumes you have SSH access to your Linux cloud server
REM Update the SSH connection details below
REM ============================================================================

echo ========================================
echo  90%% mAP@50 Target - 2 Hour Run
echo ========================================
echo.
echo Current: 86.6%% mAP@50
echo Target: 90.0%% mAP@50
echo Gap: 3.4%% (Very achievable!)
echo.

REM Option 1: Copy command to clipboard (you can paste in PuTTY/Terminal)
echo Command ready to copy:
echo.
echo cd ~/safeorbit/ok-computer ^&^& tmux new -s optimizer_90 ^&^& python scripts/model_optimizer.py --model results/improved_model/train/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.866 --final-dir ./final --strategies strategy_fast_boost strategy_quick_tune
echo.

REM Option 2: Uncomment below and update SSH details to run directly
REM SET SSH_USER=your_username
REM SET SSH_HOST=your_cloud_server.com
REM SET SSH_KEY=path\to\your\key.pem

REM ssh -i %SSH_KEY% %SSH_USER%@%SSH_HOST% "cd ~/safeorbit/ok-computer && tmux new -s optimizer_90 && python scripts/model_optimizer.py --model results/improved_model/train/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.866 --final-dir ./final --strategies strategy_fast_boost strategy_quick_tune"

pause
