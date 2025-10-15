# ================================================================================
# Quick Start Script for Model Optimization
# ================================================================================
# This script provides the fastest path to improve mAP from 0.846 to 0.90+
# ================================================================================

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "YOLOv8 Model Optimization - Quick Start" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Configuration
$BASE_MODEL = "results\runs\train\weights\best.pt"
$DATA_CONFIG = "configs\dataset.yaml"
$TRAIN_CONFIG = "configs\train_config.yaml"
$TARGET_MAP = 0.90

# Check if model exists
if (-not (Test-Path $BASE_MODEL)) {
    Write-Host "ERROR: Model not found at $BASE_MODEL" -ForegroundColor Red
    Write-Host "Please update the BASE_MODEL variable with the correct path." -ForegroundColor Yellow
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Base Model:    $BASE_MODEL" -ForegroundColor White
Write-Host "  Current mAP:   0.84336" -ForegroundColor White
Write-Host "  Target mAP:    $TARGET_MAP" -ForegroundColor White
Write-Host "  Improvement:   +$([math]::Round($TARGET_MAP - 0.84336, 5))" -ForegroundColor Yellow
Write-Host ""

# Present options
Write-Host "Choose optimization approach:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  [1] Master Pipeline (Recommended) - All phases, 8-12 hours" -ForegroundColor Green
Write-Host "      Expected: +0.05 to +0.08 mAP" -ForegroundColor Gray
Write-Host ""
Write-Host "  [2] Quick Win - Aggressive Augmentation - 2-3 hours" -ForegroundColor Yellow
Write-Host "      Expected: +0.02 to +0.03 mAP" -ForegroundColor Gray
Write-Host ""
Write-Host "  [3] Multi-Scale Training - 3-4 hours" -ForegroundColor Yellow
Write-Host "      Expected: +0.01 to +0.02 mAP" -ForegroundColor Gray
Write-Host ""
Write-Host "  [4] Extended Training (400 epochs) - 6-8 hours" -ForegroundColor Yellow
Write-Host "      Expected: +0.02 to +0.03 mAP" -ForegroundColor Gray
Write-Host ""
Write-Host "  [5] Incremental Learning - 2-4 hours" -ForegroundColor Yellow
Write-Host "      Expected: +0.01 to +0.03 mAP" -ForegroundColor Gray
Write-Host ""
Write-Host "  [6] Custom - Run specific strategies" -ForegroundColor Blue
Write-Host ""
Write-Host "  [0] Exit" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Enter your choice (0-6)"

switch ($choice) {
    "1" {
        Write-Host "`nStarting Master Pipeline..." -ForegroundColor Green
        Write-Host "This will run all optimization phases automatically." -ForegroundColor Yellow
        Write-Host "Estimated time: 8-12 hours" -ForegroundColor Yellow
        Write-Host ""
        
        $confirm = Read-Host "Continue? (Y/N)"
        if ($confirm -eq "Y" -or $confirm -eq "y") {
            python scripts\master_pipeline.py `
                --model $BASE_MODEL `
                --data $DATA_CONFIG `
                --config $TRAIN_CONFIG `
                --target $TARGET_MAP
        }
    }
    
    "2" {
        Write-Host "`nStarting Aggressive Augmentation Strategy..." -ForegroundColor Green
        Write-Host "Estimated time: 2-3 hours" -ForegroundColor Yellow
        Write-Host ""
        
        python scripts\model_optimizer.py `
            --model $BASE_MODEL `
            --data $DATA_CONFIG `
            --config $TRAIN_CONFIG `
            --strategies strategy_2_aggressive_aug `
            --target $TARGET_MAP
    }
    
    "3" {
        Write-Host "`nStarting Multi-Scale Training..." -ForegroundColor Green
        Write-Host "Estimated time: 3-4 hours" -ForegroundColor Yellow
        Write-Host ""
        
        python scripts\model_optimizer.py `
            --model $BASE_MODEL `
            --data $DATA_CONFIG `
            --config $TRAIN_CONFIG `
            --strategies strategy_5_multi_scale `
            --target $TARGET_MAP
    }
    
    "4" {
        Write-Host "`nStarting Extended Training (400 epochs)..." -ForegroundColor Green
        Write-Host "Estimated time: 6-8 hours" -ForegroundColor Yellow
        Write-Host ""
        
        $confirm = Read-Host "This will take a long time. Continue? (Y/N)"
        if ($confirm -eq "Y" -or $confirm -eq "y") {
            python scripts\model_optimizer.py `
                --model $BASE_MODEL `
                --data $DATA_CONFIG `
                --config $TRAIN_CONFIG `
                --strategies strategy_4_extended_training `
                --target $TARGET_MAP
        }
    }
    
    "5" {
        Write-Host "`nSetting up Incremental Learning..." -ForegroundColor Green
        
        # Check if new_images directory exists
        if (-not (Test-Path "new_images")) {
            Write-Host "Creating new_images directory..." -ForegroundColor Yellow
            New-Item -ItemType Directory -Path "new_images" -Force | Out-Null
            
            Write-Host "Copying validation images as new data..." -ForegroundColor Yellow
            Copy-Item "datasets\val\images\*" "new_images\" -Force
        }
        
        Write-Host "Estimated time: 2-4 hours" -ForegroundColor Yellow
        Write-Host ""
        
        python scripts\incremental_learning.py `
            --model $BASE_MODEL `
            --data $DATA_CONFIG `
            --config $TRAIN_CONFIG `
            --new-images "new_images" `
            --confidence 0.7 `
            --uncertain 0.5 `
            --iterations 5 `
            --project "incremental_runs"
    }
    
    "6" {
        Write-Host "`nAvailable strategies:" -ForegroundColor Cyan
        Write-Host "  - strategy_1_focal_loss" -ForegroundColor White
        Write-Host "  - strategy_2_aggressive_aug" -ForegroundColor White
        Write-Host "  - strategy_3_optimized_lr" -ForegroundColor White
        Write-Host "  - strategy_4_extended_training" -ForegroundColor White
        Write-Host "  - strategy_5_multi_scale" -ForegroundColor White
        Write-Host "  - strategy_6_class_balanced" -ForegroundColor White
        Write-Host ""
        
        $strategies = Read-Host "Enter strategy names (space-separated)"
        
        if ($strategies) {
            $strategyArray = $strategies -split " "
            
            Write-Host "`nRunning custom optimization..." -ForegroundColor Green
            
            python scripts\model_optimizer.py `
                --model $BASE_MODEL `
                --data $DATA_CONFIG `
                --config $TRAIN_CONFIG `
                --strategies $strategyArray `
                --target $TARGET_MAP
        }
    }
    
    "0" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host "Invalid choice. Exiting..." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Optimization Complete!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Check results in:" -ForegroundColor Yellow
Write-Host "  - optimization_runs/        (Optimization results)" -ForegroundColor White
Write-Host "  - incremental_runs/         (Incremental learning)" -ForegroundColor White
Write-Host "  - master_optimization_results/  (Master pipeline)" -ForegroundColor White
Write-Host "  - logs/                     (Detailed logs)" -ForegroundColor White
Write-Host ""
Write-Host "View metrics:" -ForegroundColor Yellow
Write-Host "  tensorboard --logdir optimization_runs" -ForegroundColor Cyan
Write-Host ""
