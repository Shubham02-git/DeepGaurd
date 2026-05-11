# DeepfakeBench Quick Start Guide

## ✅ Installation Complete!

Your system is ready:
- **GPU:** NVIDIA GeForce RTX 2050 (4GB)
- **PyTorch:** 2.6.0+cu124
- **Config:** Optimized for 4GB VRAM

---

## 📁 Download Datasets (Required)

You need deepfake datasets to train. Download one of these:

### Option 1: FaceForensics++ (Recommended)
1. Request access: https://github.com/ondyari/FaceForensics
2. Download compressed version (c23)
3. Extract to: `./datasets/FaceForensics++/`

### Option 2: Celeb-DF
1. Download: https://github.com/yuezunli/celeb-deepfakeforensics
2. Extract to: `./datasets/Celeb-DF-v2/`

### Option 3: Test with Small Dataset
Create a tiny test dataset structure:
```
datasets/
├── FaceForensics++/
│   ├── real/
│   │   └── video1.mp4
│   └── fake/
│       └── video2.mp4
```

---

## 🚀 Run Training

### Method 1: Quick Test (2 epochs)
```powershell
python training/train.py `
  --detector_path ./training/config/detector/sbi.yaml
```

### Method 2: Full Training
Edit `./training/config/detector/sbi.yaml`:
- Change `nEpochs: 2` → `nEpochs: 50`

Then run:
```powershell
python training/train.py `
  --detector_path ./training/config/detector/sbi.yaml
```

### Method 3: Custom Detector
Choose from 34 detectors in `./training/config/detector/`:
- `xception.yaml` - Classic baseline
- `efficientnetb4.yaml` - Fast & accurate
- `videomae.yaml` - Video transformer

```powershell
python training/train.py `
  --detector_path ./training/config/detector/xception.yaml
```

---

## 📊 Monitor Training

### GPU Usage (in another PowerShell window):
```powershell
nvidia-smi -l 1
```

### TensorBoard:
```powershell
tensorboard --logdir=./logs
```
Then open: http://localhost:6006

---

## 🧪 Test/Inference

After training, test on new videos:
```powershell
python training/test.py `
  --detector_path ./training/config/detector/sbi.yaml `
  --test_dataset Celeb-DF-v2
```

---

## ⚠️ If VRAM Errors Occur

Edit `./training/config/detector/sbi.yaml`:
```yaml
train_batchSize: 2  # Reduce from 4
test_batchSize: 4   # Reduce from 8
```

---

## 📝 Config Changes Made for 4GB VRAM

Original → Optimized:
- `train_batchSize: 24` → `4`
- `test_batchSize: 32` → `8`
- `workers: 8` → `2`
- `nEpochs: 50` → `2` (for testing)
- Log dir: Linux path → `./logs/sbi`
- Pretrained path: Linux → Windows

---

## 🎯 Next Steps

1. **Download dataset** (see above)
2. **Preprocess dataset:**
   ```powershell
   python preprocessing/preprocess.py `
     --dataset FaceForensics++ `
     --compression c23
   ```
3. **Run training** (see commands above)
4. **Monitor with TensorBoard**

---

## 🆘 Troubleshooting

### "CUDA out of memory"
→ Reduce batch size to 2 (or even 1)

### "Dataset not found"
→ Check dataset path in config matches your folder structure

### "Module not found"
→ Run: `pip install <missing-module>`

### Training too slow?
→ Your 4GB GPU is the bottleneck. Consider cloud GPU (Google Colab, AWS)

---

## 📚 Resources

- **Datasets:** Check preprocessing/dataset_json/readme.md

---

**Status:** ✅ Ready to train once datasets are downloaded!
