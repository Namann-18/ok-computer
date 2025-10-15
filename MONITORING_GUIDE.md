# 📊 V100 Training Monitoring Guide

## ✅ Training Started Successfully!

Your training is running in the background on V100 with **PID: 497496**

---

## 🎯 Quick Commands (Copy & Paste on V100)

### 1️⃣ Install TensorBoard (One-time)
```bash
pip install tensorboard
```

### 2️⃣ View Training Progress (Real-time)
```bash
# See latest log entries
tail -f training_v100.log

# Exit with: Ctrl+C
```

### 3️⃣ Check Training Metrics
```bash
# Quick status check
chmod +x monitor_training.sh
./monitor_training.sh

# Search for mAP scores
cat training_v100.log | grep "mAP50"

# Search for losses
cat training_v100.log | grep "box_loss\|cls_loss"
```

### 4️⃣ Monitor GPU Utilization
```bash
# Real-time GPU monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Exit with: Ctrl+C

# Or single check
nvidia-smi
```

### 5️⃣ Check Training Process
```bash
# See if training is running
ps aux | grep train.py

# Check CPU/Memory usage
top -p 497496

# Exit with: q
```

### 6️⃣ Start TensorBoard (After Installation)
```bash
# Start TensorBoard (access from browser)
tensorboard --logdir results/runs --host 0.0.0.0 --port 6006

# Access at: http://<your-vm-ip>:6006
# You may need to configure firewall rules in Google Cloud
```

---

## ⏱️ Timeline & Expectations

| Time Elapsed | Expected Epoch | Expected mAP@50 |
|--------------|----------------|-----------------|
| 0-1 hour | 0-30 | Initializing |
| 1-2 hours | 30-60 | 78-82% |
| 3-4 hours | 60-120 | 82-86% |
| 6-8 hours | 120-200 | 86-89% |
| 10-12 hours | 200-300 | 89-91% |
| **12-16 hours** | **300-350** | **90-92%** ✅ |

### What to Watch For

**Good Signs ✅**
- GPU utilization: 90-100%
- Memory usage: 12-15GB / 16GB
- mAP@50 steadily increasing
- Loss values decreasing
- Process still running

**Warning Signs ⚠️**
- GPU utilization < 50%
- Process not found
- Out of memory errors
- Loss not decreasing after 50 epochs

---

## 📈 Reading the Training Log

### Key Metrics to Look For

```bash
# Example log output
Epoch 100/350: 
  box_loss: 0.842      # Should decrease (target: < 1.0)
  cls_loss: 0.456      # Should decrease (target: < 0.5)
  dfl_loss: 0.912      # Should decrease
  
Validation Results:
  mAP50: 0.876         # Should increase (target: > 0.90)
  mAP50-95: 0.642      # Should increase (target: > 0.70)
  Precision: 0.889     # Should increase (target: > 0.90)
  Recall: 0.834        # Should increase (target: > 0.85)
```

### What Good Progress Looks Like

```
Epoch   10/350: mAP50=0.720 (initializing)
Epoch   50/350: mAP50=0.810 (learning)
Epoch  100/350: mAP50=0.850 (improving)
Epoch  150/350: mAP50=0.880 (converging)
Epoch  200/350: mAP50=0.895 (fine-tuning)
Epoch  250/350: mAP50=0.905 (polishing)
Epoch  300/350: mAP50=0.910 (near target)
Epoch  350/350: mAP50=0.915 (complete!) ✅
```

---

## 🔍 Detailed Monitoring Script

Run this for comprehensive status:

```bash
# Make executable
chmod +x monitor_training.sh

# Run monitoring
./monitor_training.sh
```

**Output includes:**
- ✅ Process status (running/stopped)
- 📊 GPU utilization, memory, temperature
- 📈 Current epoch and metrics
- ⏱️ Estimated time remaining
- 💾 Disk space
- 📁 Latest checkpoints

---

## 🛑 If You Need to Stop Training

```bash
# Find process ID
ps aux | grep train.py

# Stop gracefully (recommended)
kill 497496

# Force stop (if needed)
kill -9 497496

# Resume later from checkpoint
python scripts/train.py --config configs/train_config.yaml --resume results/runs/train/weights/last.pt
```

---

## 🔧 Troubleshooting

### Issue: "Training seems stuck"

**Check:**
```bash
# Is process running?
ps aux | grep train.py

# GPU activity?
nvidia-smi

# Log file updating?
ls -lh training_v100.log
tail training_v100.log
```

### Issue: "Out of memory"

**Solution:**
```bash
# Stop training
kill 497496

# Reduce batch size in config
nano configs/train_config.yaml
# Change: batch_size: 32 → batch_size: 24

# Restart training
./train_on_v100.sh
```

### Issue: "Process died"

**Check log for errors:**
```bash
tail -100 training_v100.log

# Look for:
# - CUDA out of memory
# - File not found
# - Import errors
```

**Resume from checkpoint:**
```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --data configs/dataset.yaml \
    --resume results/runs/train/weights/last.pt
```

---

## 📊 TensorBoard Setup (Optional but Recommended)

### Install TensorBoard
```bash
pip install tensorboard
```

### Start TensorBoard
```bash
tensorboard --logdir results/runs --host 0.0.0.0 --port 6006
```

### Access TensorBoard

1. **Option A: SSH Tunnel (Recommended)**
   ```bash
   # On your local machine:
   ssh -L 6006:localhost:6006 namannayak_16@<vm-ip>
   
   # Then open: http://localhost:6006
   ```

2. **Option B: Firewall Rule**
   ```bash
   # In Google Cloud Console:
   # VPC Network → Firewall → Create Rule
   # - Target: All instances
   # - Source IP: Your IP / 0.0.0.0/0
   # - Protocols/Ports: tcp:6006
   
   # Then access: http://<vm-external-ip>:6006
   ```

3. **Option C: ngrok (Quick & Easy)**
   ```bash
   # Install ngrok
   wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
   tar -xvzf ngrok-v3-stable-linux-amd64.tgz
   
   # Start tunnel
   ./ngrok http 6006
   
   # Use the ngrok URL provided
   ```

---

## ✅ Success Indicators

Training is going well if:

1. **Process Running**: ✅ `ps aux | grep train.py` shows process
2. **GPU Utilized**: ✅ `nvidia-smi` shows 90-100% GPU usage
3. **Log Growing**: ✅ `ls -lh training_v100.log` file size increasing
4. **Metrics Improving**: ✅ mAP@50 increasing, losses decreasing
5. **No Errors**: ✅ No "error", "exception", "failed" in logs

---

## 📅 What Happens Next?

### During Training (Next 12-16 hours)
- Training runs automatically in background
- Checkpoints saved every 10 epochs
- Best model saved automatically when mAP improves
- You can safely disconnect from SSH

### After Training Completes
1. **Check final results**
   ```bash
   cat training_v100.log | tail -50
   ```

2. **Find best model**
   ```bash
   ls -lh results/runs/train/weights/best.pt
   # or
   ls -lh models/best.pt
   ```

3. **Test on real images**
   ```bash
   python scripts/inference_tta.py \
       --model results/runs/train/weights/best.pt \
       --source path/to/real/images \
       --output results/real_test
   ```

4. **Update API**
   - Edit `api.py`: Update MODEL_PATH
   - Restart API server

---

## 🎉 Expected Final Results

```
Training Summary (Epoch 350/350)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metrics:
  mAP@50:     0.915  ✅ (target: 0.90)
  mAP@50-95:  0.703  ✅ (target: 0.70)
  Precision:  0.922  ✅ (target: 0.90)
  Recall:     0.867  ✅ (target: 0.85)

Training Time: 14.2 hours
Best Model: results/runs/train/weights/best.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Model ready for real-world deployment!
```

---

## 💡 Pro Tips

1. **Safe to Disconnect**: Training runs with `nohup`, won't stop if you disconnect
2. **Use tmux**: Even safer - training survives SSH disconnection
   ```bash
   tmux new -s training
   ./train_on_v100.sh
   # Ctrl+B, D to detach
   # tmux attach -t training (to reattach)
   ```
3. **Check Periodically**: Monitor every 2-4 hours
4. **Don't Stop Early**: Let it complete all 350 epochs for best results
5. **Backups**: Models auto-saved to `results/runs/train/weights/`

---

## 📞 Quick Reference

```bash
# Training status
./monitor_training.sh

# Live log
tail -f training_v100.log

# GPU status
nvidia-smi

# Process status
ps aux | grep train.py

# Latest metrics
cat training_v100.log | grep "mAP50" | tail -5

# TensorBoard
tensorboard --logdir results/runs --host 0.0.0.0 --port 6006
```

---

**Your training is running! Check back in 12-16 hours for completion.** ✨🚀

**Next**: Read through this guide and set up monitoring tools. Your model will be ready for real-world images tomorrow!
