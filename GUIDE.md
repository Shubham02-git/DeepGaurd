# DeepGuard — Complete Setup & Operations Guide

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Training Status](#2-current-training-status)
3. [Check Training Progress (Live)](#3-check-training-progress-live)
4. [Start the Auto-Sequence (UCF → I3D → EfficientNetB4)](#4-start-the-auto-sequence)
5. [What To Do When Training Finishes](#5-what-to-do-when-training-finishes)
6. [Start the Website](#6-start-the-website)
7. [Full From-Scratch Setup Guide](#7-full-from-scratch-setup-guide)
8. [Folder Structure Reference](#8-folder-structure-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Project Overview

DeepGuard is a deepfake detection system for AI-powered fake media detection.

| Component | Location | Purpose |
|-----------|----------|---------|
| Training pipeline | `training/` | Trains detection models |
| Inference API | `inference_api.py` | FastAPI server, serves predictions |
| Frontend | `frontend/` | React UI for uploading images/videos |
| Dataset | `datasets/rgb/` | Extracted face frames (DFD dataset) |
| Checkpoints | `logs/training/` | Saved model weights post-training |
| Configs | `training/config/detector/` | YAML configs per model |

**Active Python environment:** `.venv/` (contains PyTorch + CUDA — do NOT delete)

---

## 2. Current Training Status

| Order | Model | Config | Status | Est. Time |
|-------|-------|--------|--------|-----------|
| 1 | **Xception** | `xception_dfd.yaml` | 🟢 Running | ~7.7 hrs total |
| 2 | **EfficientNetB4** | `efficientnetb4_dfd.yaml` | ⏳ Queued | ~5-6 hrs (resumes epoch 2) |
| 3 | **UCF** | `ucf_dfd.yaml` | ⏳ Queued | ~6-8 hrs |
| 4 | **I3D** | `i3d_dfd.yaml` | ⏳ Queued | ~10-14 hrs |

Checkpoints are saved to `logs/training/<model>_<timestamp>/test/avg/ckpt_best.pth`

---

## 3. Check Training Progress (Live)

### See what's currently running
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, WorkingSet
```

### Watch the live training log (most recent run)
```powershell
# Get the latest log file
$latest = Get-ChildItem "logs/training" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content "$($latest.FullName)\training.log" -Wait
```

### Get last 30 lines of latest training log
```powershell
$latest = Get-ChildItem "logs/training" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content "$($latest.FullName)\training.log" -Tail 30
```

### List all training runs with timestamps
```powershell
Get-ChildItem "logs/training" | Select-Object Name, LastWriteTime | Sort-Object LastWriteTime -Descending
```

### Check if a best checkpoint exists (training has completed at least 1 epoch)
```powershell
Get-ChildItem "logs/training" -Recurse -Filter "ckpt_best.pth" | Select-Object FullName, LastWriteTime
```

### Check GPU usage (requires nvidia-smi)
```powershell
nvidia-smi
```

### Check if Xception is still training
```powershell
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*xception*" }
```

---

## 4. Start the Auto-Sequence

The sequence script waits for Xception to finish, then automatically runs:
**EfficientNetB4 → UCF → I3D**

### Open a NEW terminal and run:
```powershell
cd "f:\repo's\DeepfakeBench-main\DeepfakeBench-main"
.\run_training_sequence.ps1
```

The script will:
1. Poll every 60 seconds for the Xception process to exit
2. Print status updates with timestamps
3. Start each model in sequence, stopping if any fails
4. Print completion time for each model

> **Note:** Keep this terminal open. If you close it, the sequence stops.  
> Xception must still be running when you start this — the script just waits for it.

### To run a single model manually (any time):
```powershell
# Activate environment first
.\.venv\Scripts\Activate.ps1

# Then run any model
.\.venv\Scripts\python.exe training/train.py --detector_path training/config/detector/xception_dfd.yaml
.\.venv\Scripts\python.exe training/train.py --detector_path training/config/detector/efficientnetb4_dfd.yaml
.\.venv\Scripts\python.exe training/train.py --detector_path training/config/detector/ucf_dfd.yaml
.\.venv\Scripts\python.exe training/train.py --detector_path training/config/detector/i3d_dfd.yaml
```

---

## 5. What To Do When Training Finishes

### Step 1 — Find all best checkpoints
```powershell
Get-ChildItem "logs/training" -Recurse -Filter "ckpt_best.pth" | Select-Object FullName, LastWriteTime, @{N='Size_KB';E={[math]::Round($_.Length/1KB,1)}}
```

### Step 2 — Compare AUC scores
Each training run saves a `metric_dict_best.pickle` alongside the checkpoint.
To print the AUC for each model, run:
```powershell
.\.venv\Scripts\python.exe -c "
import pickle, glob, os
for p in glob.glob('logs/training/*/test/avg/metric_dict_best.pickle'):
    model = p.split(os.sep)[2]
    with open(p, 'rb') as f:
        d = pickle.load(f)
    auc = d.get('auc', d.get('video_auc', 'N/A'))
    print(f'{model:50s}  AUC={auc}')
"
```

### Step 3 — The inference API auto-discovers checkpoints
No manual wiring is needed. `inference_api.py` automatically scans `logs/training/`
and loads the **most recent** `ckpt_best.pth` per model at startup.

To verify what it will load:
```powershell
.\.venv\Scripts\python.exe -c "
import os, json

LOGS = 'logs/training'
for key in ['xception', 'ucf', 'efficientnetb4']:
    candidates = sorted([d for d in os.listdir(LOGS) if d.startswith(key + '_')], reverse=True)
    found = None
    for folder in candidates:
        ckpt = os.path.join(LOGS, folder, 'test', 'avg', 'ckpt_best.pth')
        if os.path.exists(ckpt):
            found = ckpt
            break
    print(f'{key:20s} -> {found or \"NOT FOUND\"}')
"
```

### Step 4 — Start the website (see Section 6)

---

## 6. Start the Website

### One-command start (API + Frontend together):
```powershell
cd "f:\repo's\DeepfakeBench-main\DeepfakeBench-main"
.\start.ps1
```

This opens two terminal windows:
- **API** at `http://localhost:8000`
- **UI** at `http://localhost:5173`

---

### Or start them separately:

#### Terminal 1 — Inference API
```powershell
cd "f:\repo's\DeepfakeBench-main\DeepfakeBench-main"
.\.venv\Scripts\Activate.ps1
python -m uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 — React Frontend
```powershell
cd "f:\repo's\DeepfakeBench-main\DeepfakeBench-main\frontend"
npm run dev
```

---

### Verify API is working:
```powershell
# Health check
Invoke-RestMethod http://localhost:8000/health

# See which models are loaded
Invoke-RestMethod http://localhost:8000/models

# Check device (CPU/CUDA)
Invoke-RestMethod http://localhost:8000/
```

### Test prediction from command line:
```powershell
# Replace path with any image
curl -X POST "http://localhost:8000/predict?model=xception" -F "file=@C:\path\to\image.jpg"
```

---

### API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Status + which models are loaded |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List all models + checkpoint paths |
| `POST` | `/predict?model=xception` | Upload image/video → get prediction |

**Model options for `?model=`:** `xception` (default), `ucf`, `efficientnetb4`

---

## 7. Full From-Scratch Setup Guide

> Follow this if you're setting up on a new machine or after a full reset.

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (RTX recommended)
- Node.js 18+ and npm
- ~50 GB free disk space

---

### Step 1 — Enter the project directory
```powershell
cd DeepfakeBench-main
```

### Step 2 — Create Python environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install Python dependencies (PyTorch with CUDA)
```powershell
# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Other dependencies
pip install fastapi uvicorn python-multipart opencv-python pillow pyyaml
pip install scikit-learn scikit-image tensorboard timm einops
pip install albumentations efficientnet_pytorch ftfy regex
```

### Step 4 — Install frontend dependencies
```powershell
cd frontend
npm install
cd ..
```

### Step 5 — Download pretrained weights
Place these files in `training/pretrained/pretrained/`:
- `xception-b5690688.pth` — [download from Google Drive](https://drive.google.com/file/d/1H0gsSvfz-2T3Bp33bK6W2dVw-6W9AZWM/view)
- `efficientnet-b4-6ed6700e.pth` — downloaded automatically by `efficientnet_pytorch`

Place this file in `training/pretrained/`:
- `I3D_8x8_R50.pth` — download from official model repository

### Step 6 — Prepare dataset (DFD)
```powershell
# Extract face frames from DFD videos
.\.venv\Scripts\python.exe preprocessing/preprocess_dfd.py

# Generate train/val/test split JSON
.\.venv\Scripts\python.exe preprocessing/generate_dfd_json.py
```

### Step 7 — Train models (run in separate terminals or use sequence script)
```powershell
# Terminal A — Xception (start first, fastest to converge)
.\.venv\Scripts\python.exe training/train.py --detector_path training/config/detector/xception_dfd.yaml

# Terminal B — Run this WHILE xception is training; it auto-chains the rest
.\run_training_sequence.ps1
```

### Step 8 — Start the website
```powershell
.\start.ps1
```

Open `http://localhost:5173` in your browser.

---

## 8. Folder Structure Reference

```
DeepfakeBench-main/
├── .venv/                          ← Active Python env (PyTorch + CUDA)
├── inference_api.py                ← FastAPI server
├── start.ps1                       ← One-command launcher
├── run_training_sequence.ps1       ← Auto-chains EfficientNetB4 → UCF → I3D
├── GUIDE.md                        ← This file
│
├── frontend/                       ← React UI
│   └── src/
│       ├── App.jsx                 ← Main app + API calls
│       └── components/
│           ├── UploadZone.jsx      ← File upload + model selector
│           └── ResultCard.jsx      ← Detection result display
│
├── training/
│   ├── train.py                    ← Training entry point
│   ├── config/detector/
│   │   ├── xception_dfd.yaml       ← Xception config (DFD)
│   │   ├── efficientnetb4_dfd.yaml ← EfficientNetB4 config (DFD, resumes epoch 2)
│   │   ├── ucf_dfd.yaml            ← UCF config (DFD)
│   │   └── i3d_dfd.yaml            ← I3D config (DFD, video model)
│   └── pretrained/
│       ├── pretrained/
│       │   └── xception-b5690688.pth
│       └── I3D_8x8_R50.pth
│
├── preprocessing/
│   ├── preprocess_dfd.py           ← Face extraction from DFD videos
│   ├── generate_dfd_json.py        ← Generates DFD.json split file
│   └── dataset_json/
│       └── DFD.json                ← Train/val/test split (relative paths)
│
├── datasets/
│   └── rgb/                        ← Extracted face frames (DO NOT DELETE)
│
└── logs/
    └── training/
        ├── xception_<timestamp>/
        │   └── test/avg/ckpt_best.pth   ← Best Xception checkpoint
        ├── efficientnetb4_<timestamp>/
        │   └── test/avg/ckpt_best.pth
        ├── ucf_<timestamp>/
        │   └── test/avg/ckpt_best.pth
        └── i3d_<timestamp>/
            └── test/avg/ckpt_best.pth
```

---

## 9. Troubleshooting

### Training crashes immediately
```powershell
# Check the training log for the error
$latest = Get-ChildItem "logs/training" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content "$($latest.FullName)\training.log" -Tail 50
```

### "CUDA out of memory"
Reduce batch size in the relevant config YAML:
```yaml
train_batchSize: 16   # try halving this
```

### API won't start — "Model not found"
Training hasn't finished yet. The API will still start but show models as `"loaded": false`.
Run `Invoke-RestMethod http://localhost:8000/models` to check status.

### API starts but checkpoint not found for a model
```powershell
# Check if the checkpoint actually exists
Get-ChildItem "logs/training" -Recurse -Filter "ckpt_best.pth"
```
If missing, the model hasn't finished training or crashed before saving epoch 1.

### Frontend can't reach API (CORS / connection refused)
- Make sure API is running: `Invoke-RestMethod http://localhost:8000/health`
- Check port isn't blocked: API runs on `8000`, frontend on `5173`
- CORS is set to `allow_origins=["*"]` so any origin is allowed

### EfficientNetB4 resume fails
The resume checkpoint path is hardcoded in `efficientnetb4_dfd.yaml`:
```yaml
resume: ./logs/training/efficientnetb4_2026-03-01-17-42-51/test/avg/ckpt_best.pth
```
If that file doesn't exist, remove the `resume:` line and set `start_epoch: 0` to train from scratch.

### Check PyTorch + CUDA is working
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

### Re-run preprocessing (if frames missing)
```powershell
.\.venv\Scripts\python.exe preprocessing/preprocess_dfd.py
```
The script skips already-processed videos, so it's safe to re-run.
