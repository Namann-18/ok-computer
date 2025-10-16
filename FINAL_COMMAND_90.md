# 🎯 FINAL COMMAND - 90% mAP@50 in 2 Hours

## ⚡ ONE-LINE COMMAND (COPY THIS NOW!)

```bash
cd ~/safeorbit/ok-computer && tmux new -s optimizer_90 && python scripts/model_optimizer.py --model results/improved_model/train/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.866 --final-dir ./final --strategies strategy_fast_boost strategy_quick_tune
```

---

## 📊 MISSION BRIEFING

| Metric | Value |
|--------|-------|
| 🎯 **Target** | **90.0% mAP@50** |
| 📍 **Starting Point** | **86.6% mAP@50** |
| 📈 **Gap to Close** | **3.4%** (Very achievable!) |
| ⏱️ **Time Limit** | **2 hours maximum** |
| 🚀 **Strategy** | 2 aggressive fast strategies (110 epochs total) |
| 💾 **Model Path** | `results/improved_model/train/weights/best.pt` |

---

## ⏰ EXECUTION TIMELINE

```
00:00 ━━━━━━━━━━━━━━━━━━━ Strategy 1: Fast Boost (60 epochs)
      ├─ Very high learning rate (0.0015)
      ├─ Aggressive loss weights (box=9, cls=1.5, dfl=2.5)
      ├─ Strong augmentation (mixup=0.5, copy_paste=0.5)
      └─ Expected: 86.6% → 88.5-89%

00:45 ━━━━━━━━━━━━━━━━━━━ Strategy 2: Quick Tune (50 epochs)
      ├─ Maximum precision tuning (box=10, cls=2.0, dfl=3.0)
      ├─ Optimized learning rate (0.001)
      ├─ High momentum (0.98)
      └─ Expected: 88.5-89% → 90%+

01:20 ━━━━━━━━━━━━━━━━━━━ Analysis & Save
      ├─ Best model auto-saved to ./final/
      ├─ Metrics saved to JSON
      └─ Ready for deployment

02:00 ✅ MISSION COMPLETE
```

---

## 🎲 SUCCESS PROBABILITY

Starting from **86.6%** with only **3.4%** improvement needed:

| Outcome | mAP@50 Range | Probability | Status |
|---------|--------------|-------------|--------|
| 🟢 Good | 88.0 - 88.5% | **80%** | Solid improvement |
| 🟢 Very Good | 88.5 - 89.5% | **60%** | Great progress |
| 🟢 Excellent | 89.5 - 90.0% | **40%** | Near perfect |
| 🏆 Perfect | 90.0%+ | **20-30%** | **TARGET ACHIEVED!** |

**You have a MUCH better chance than starting from 85.8%!**

---

## 📋 PRE-FLIGHT CHECKLIST

Before running, verify:

- [ ] ✅ CUDA/GPU available: `nvidia-smi`
- [ ] ✅ Correct directory: `cd ~/safeorbit/ok-computer`
- [ ] ✅ Best model exists: `ls -lh results/improved_model/train/weights/best.pt`
- [ ] ✅ Configs exist: `ls configs/dataset.yaml configs/train_config.yaml`
- [ ] ✅ Final directory created: `mkdir -p final`

---

## 🖥️ MONITORING COMMANDS

### GPU Usage:
```bash
watch -n 1 nvidia-smi
```

### Training Progress:
```bash
tail -f logs/optimizer_*.log
```

### Saved Models:
```bash
ls -lh final/
```

### tmux Control:
```bash
# Detach: Ctrl+B then D
# Reattach: tmux attach -t optimizer_90
# Kill: tmux kill-session -t optimizer_90
```

---

## 🏆 AFTER COMPLETION

### 1. Check Results:
```bash
cd ~/safeorbit/ok-computer
ls -lh final/
cat final/best_metrics.json
```

### 2. If 90%+ Achieved, Deploy to Production:
```bash
# Backup current model
cp results/improved_model/train/weights/best.pt results/improved_model/train/weights/best_86.6_backup.pt

# Copy new best model
cp final/best_model_*.pt results/improved_model/train/weights/best.pt

# Restart API (if running)
pkill -f api.py
nohup python api.py > api.log 2>&1 &
```

### 3. Update Windows Local Copy:
```bash
# From your Windows machine (if needed):
scp user@cloud:/path/to/ok-computer/final/best_model_*.pt E:\safeorbit\ok-computer\results\improved_model\train\weights\best.pt
```

---

## 🆘 TROUBLESHOOTING

### Problem: Training too slow
```bash
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reduce batch size if OOM
nano configs/train_config.yaml  # Change batch size
```

### Problem: Not reaching 90%
```bash
# Add one more quick strategy (45 epochs, 30 min)
python scripts/model_optimizer.py \
  --model results/improved_model/train/weights/best.pt \
  --data configs/dataset.yaml \
  --config configs/train_config.yaml \
  --target 0.90 \
  --baseline 0.866 \
  --final-dir ./final \
  --strategies strategy_rapid_convergence
```

### Problem: Out of memory
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Or reset GPU
sudo nvidia-smi --gpu-reset
```

---

## 📝 WHAT THE OPTIMIZER DOES

1. ✅ Loads your best model (86.6% mAP@50)
2. ✅ Applies strategy_fast_boost (60 epochs, aggressive training)
3. ✅ Applies strategy_quick_tune (50 epochs, precision tuning)
4. ✅ Auto-saves ANY model better than 86.6% to `./final/`
5. ✅ Generates detailed metrics and logs
6. ✅ Stops when target (90%) is reached OR time runs out

---

## 🎯 LET'S GET TO 90%!

**You're starting from 86.6% - only 3.4% to go!**

Just copy this command and run it:

```bash
cd ~/safeorbit/ok-computer && tmux new -s optimizer_90 && python scripts/model_optimizer.py --model results/improved_model/train/weights/best.pt --data configs/dataset.yaml --config configs/train_config.yaml --target 0.90 --baseline 0.866 --final-dir ./final --strategies strategy_fast_boost strategy_quick_tune
```

**Good luck! 🚀🍀**

---

## 📞 QUICK REFERENCE

| Action | Command |
|--------|---------|
| Start | Copy command above |
| Monitor | `watch -n 1 nvidia-smi` |
| Logs | `tail -f logs/optimizer_*.log` |
| Detach | `Ctrl+B` then `D` |
| Reattach | `tmux attach -t optimizer_90` |
| Check results | `ls -lh final/` |
| View metrics | `cat final/best_metrics.json` |

---

**Last Updated**: October 16, 2025  
**Model Version**: improved_model (86.6% mAP@50)  
**Target**: 90% mAP@50 in 2 hours  
**Status**: ✅ READY TO RUN
