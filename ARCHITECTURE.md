# DeepGuard — Project Architecture & Workflow

> A complete explainer of how DeepGuard works, from raw video to deepfake detection — covering data pipeline, model training, API, and frontend. Suitable for presenting to others.

---

## Table of Contents

1. [What is DeepGuard?](#1-what-is-deepguard)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Full Data & Inference Workflow](#3-full-data--inference-workflow)
4. [Data Pipeline (Training Side)](#4-data-pipeline-training-side)
5. [AI Models — How Each One Works](#5-ai-models--how-each-one-works)
6. [Backend (Inference API)](#6-backend-inference-api)
7. [Frontend (Web UI)](#7-frontend-web-ui)
8. [Storage & Data Layer](#8-storage--data-layer)
9. [Complete Tech Stack](#9-complete-tech-stack)
10. [How a Prediction is Made (Step-by-Step)](#10-how-a-prediction-is-made-step-by-step)

---

## 1. What is DeepGuard?

DeepGuard is an **AI-powered deepfake detection system**. Users upload an image or video through a web interface. The system runs the file through a trained neural network and returns a verdict: **Real** or **Fake**, with a confidence percentage.

It is trained on the **Deep Fake Detection (DFD)** dataset — a dataset of 3,431 real and AI-manipulated face videos.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER'S BROWSER                         │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              React Frontend  (Vite + Tailwind)           │  │
│   │   Upload image/video  →  Select model  →  View result    │  │
│   └──────────────────────────┬───────────────────────────────┘  │
│                               │  HTTP POST /predict?model=...    │
└───────────────────────────────┼─────────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   FastAPI (Python)      │
                    │   inference_api.py      │
                    │   Port 8000             │
                    └───────────┬────────────┘
                                │
               ┌────────────────┼───────────────────┐
               │                │                   │
    ┌──────────▼──┐   ┌─────────▼──┐   ┌───────────▼──┐
    │  Xception   │   │    UCF     │   │ EfficientNetB4│
    │  Model      │   │  Model     │   │   Model       │
    │  (default)  │   │            │   │               │
    └──────────┬──┘   └─────────┬──┘   └───────────┬──┘
               │                │                   │
               └────────────────┼───────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Checkpoint Files      │
                    │   logs/training/        │
                    │   */test/avg/           │
                    │   ckpt_best.pth         │
                    └──────────────────────┘
```

---

## 3. Full Data & Inference Workflow

### A) Training Workflow (done once, offline)

```
Raw DFD Videos (3,431 MP4 files)
         │
         ▼
 Face Detection & Extraction
 (OpenCV Haar Cascade)
 preprocessing/preprocess_dfd.py
         │
         ▼
 Face Frame Images (PNG)
 datasets/rgb/
 DFD_fake/  →  2,455 train / 306 val / 308 test videos
 DFD_real/  →    291 train /  36 val /  37 test videos
         │
         ▼
 Dataset Split JSON
 preprocessing/dataset_json/DFD.json
 (train 80% / val 10% / test 10%)
         │
         ▼
 Model Training
 training/train.py
 (PyTorch, CUDA GPU)
         │
         ▼
 Saved Checkpoints
 logs/training/<model>_<timestamp>/
 test/avg/ckpt_best.pth
```

---

### B) Inference Workflow (runs live on the web)

```
User uploads file (browser)
         │
         ▼
 React Frontend
 FormData POST → http://localhost:8000/predict?model=xception
         │
         ▼
 FastAPI Server (inference_api.py)
         │
         ├── If IMAGE:
         │       PIL open → Resize 256×256
         │       Normalize [-1, 1]
         │       Tensor shape: [1, 3, 256, 256]
         │
         └── If VIDEO:
                 OpenCV read → sample 16 evenly-spaced frames
                 Each frame: Resize → Normalize
                 Tensor shape: [16, 3, 256, 256]
         │
         ▼
 Selected Neural Network (Xception / UCF / EfficientNetB4)
         │
         ▼
 Forward Pass
 output logits shape: [N, 2]  (N = frames or 1)
         │
         ▼
 Softmax → [prob_real, prob_fake]
 Average across frames (videos)
         │
         ▼
 JSON Response:
 {
   "label": "fake",
   "confidence": 94.7,
   "prob_real": 5.3,
   "prob_fake": 94.7,
   "model": "xception",
   "file_type": "video",
   "frames_analyzed": 16
 }
         │
         ▼
 Frontend renders ResultCard
 (verdict, confidence bar, probability breakdown)
```

---

## 4. Data Pipeline (Training Side)

### Dataset: Deep Fake Detection (DFD)
| Split | Fake videos | Real videos |
|-------|------------|------------|
| Train | 2,455 | 291 |
| Val   | 306   | 36  |
| Test  | 308   | 37  |
| **Total** | **3,069** | **364** |

### Step 1 — Face Extraction (`preprocess_dfd.py`)
- Opens each video with **OpenCV VideoCapture**
- Extracts 32 evenly-spaced frames per video
- Detects face region using **Haar Cascade** classifier
- Crops and saves face as 256×256 PNG to `datasets/rgb/`
- Skips already-processed videos (idempotent)

### Step 2 — Dataset JSON (`generate_dfd_json.py`)
- Scans all extracted frame folders
- Splits 80% train / 10% val / 10% test (random, seeded)
- Writes `DFD.json` — a lookup table mapping video name → list of frame paths

### Step 3 — Data Augmentation (at training time)
Applied to training frames only:
| Augmentation | Probability | Purpose |
|---|---|---|
| Horizontal flip | 50% | Mirror invariance |
| Gaussian blur | 50% | Compression robustness |
| Brightness/contrast jitter | 50% | Lighting invariance |
| JPEG quality noise | 40–100% | Compression artifact sim |
| Normalization | Always | Mean=0.5, Std=0.5 → [-1,1] |

---

## 5. AI Models — How Each One Works

### Model 1 — Xception ⭐ (Default, Best)

| Property | Value |
|---|---|
| Architecture | Depthwise separable convolutions (Xception) |
| Input | 256×256 RGB image |
| Pretrained on | ImageNet |
| Fine-tuned on | DFD (10 epochs) |
| Batch size | 32 |
| Optimizer | Adam (lr=0.0002) |
| Loss | Cross-Entropy |
| Output | 2-class softmax [real, fake] |
| Expected AUC | 0.90–0.95 |

**How it detects fakes:**
Xception was designed for image classification but is exceptionally good at detecting subtle texture inconsistencies left by deepfake generation (GAN artifacts, blending boundaries, unnatural skin texture). Each "Xception block" uses depthwise separable convolutions — meaning it separately analyzes spatial patterns and cross-channel patterns, making it sensitive to fine-grained pixel anomalies that deepfakes typically introduce.

```
Input Image (256×256×3)
    │
    ▼
Entry Flow (Conv layers, stride-2 downsampling)
    │
    ▼
Middle Flow (8× repeated Xception blocks)
    │
    ▼
Exit Flow (feature pyramid)
    │
    ▼
Global Average Pooling → [2048-dim vector]
    │
    ▼
Fully Connected → [2 classes: real / fake]
    │
    ▼
Softmax → probabilities
```

---

### Model 2 — UCF (Uncovering Common Features)

| Property | Value |
|---|---|
| Architecture | Xception backbone + disentanglement heads |
| Input | 256×256 RGB image |
| Pretrained on | ImageNet (Xception weights) |
| Backbone mode | `adjust_channel` (feature width adaptation) |
| Batch size | 16 |
| Loss | 4-way multi-task loss |
| Expected AUC | 0.88–0.93 |

**How it detects fakes:**
UCF goes beyond binary classification. It learns to **disentangle** the forgery-specific features from content features, training with 4 simultaneous loss terms:

| Loss component | Purpose |
|---|---|
| `cls_loss` (cross-entropy) | Main real/fake classification |
| `spe_loss` (cross-entropy) | Forgery-type-specific features |
| `con_loss` (contrastive) | Pull same-class features together, push different apart |
| `rec_loss` (L1) | Reconstruct the content, ensuring forgery features are isolated |

This multi-task setup forces the model to build robust, generalizable deepfake representations rather than overfitting to one manipulation type.

```
Input Image
    │
    ▼
Xception Backbone (shared encoder)
    │
    ├── Classification Head → real/fake probability
    ├── Specialization Head → forgery-type features
    ├── Contrastive Head → embedding space regularization
    └── Reconstruction Head → content reconstruction
         │
         Combined weighted loss ← trains all heads jointly
```

---

### Model 3 — EfficientNetB4

| Property | Value |
|---|---|
| Architecture | EfficientNet-B4 (compound scaling) |
| Input | 256×256 RGB image |
| Pretrained on | ImageNet |
| Fine-tuned on | DFD (resumes from epoch 2) |
| Batch size | 32 |
| Loss | Cross-Entropy |
| Expected AUC | 0.85–0.90 |

**How it detects fakes:**
EfficientNet uses **compound scaling** — simultaneously scaling width, depth, and resolution of the network using a fixed ratio. B4 is the 4th scaling step, offering a good accuracy/compute tradeoff. Its MBConv blocks (mobile inverted bottleneck convolutions) are efficient at capturing multi-scale spatial features, making it effective at catching inconsistencies at various scales (from pixel-level artifacts to face-level anomalies).

```
Input Image (256×256×3)
    │
    ▼
Stem Conv (3×3, stride 2)
    │
    ▼
MBConv Blocks × 7 stages (compound-scaled)
    │
    ▼
Head Conv → Global Average Pool → [1792-dim]
    │
    ▼
Dropout → FC(2) → Softmax
```

---

### Model 4 — I3D (Inflated 3D ConvNet) — Video-native

| Property | Value |
|---|---|
| Architecture | Inflated 3D ResNet-50 |
| Input | 32 frames × 224×224 RGB (video clip) |
| Pretrained on | Kinetics-400 (human action recognition) |
| Batch size | 8 |
| Loss | Cross-Entropy |
| Special | Only model that processes temporal motion between frames |

**How it detects fakes:**
I3D "inflates" 2D image convolutions into 3D space-time convolutions — meaning it looks at **N frames simultaneously**, detecting inconsistencies in how a face moves over time. Deepfakes often have:
- Temporal flickering (frame-to-frame inconsistency)
- Unnatural blinking or lip sync
- Motion blur mismatch between face and background

I3D captures these temporal signals that single-frame models (Xception, UCF, EfficientNetB4) miss entirely.

```
Video (32 frames × 224×224)
    │
    ▼
3D Conv layers (filter shape: T×H×W)
    │
    ▼
Inflated ResNet-50 blocks
(spatial + temporal feature maps)
    │
    ▼
3D Global Average Pool → [2048-dim]
    │
    ▼
FC(2) → Softmax [real / fake]
```

---

### Model Comparison

| Model | Input | Temporal? | Speed | Expected AUC | Best for |
|---|---|---|---|---|---|
| Xception | Single frame | No | Fast | 0.90–0.95 | General use, images |
| UCF | Single frame | No | Medium | 0.88–0.93 | Cross-dataset generalization |
| EfficientNetB4 | Single frame | No | Fast | 0.85–0.90 | Lightweight deployment |
| I3D | 32-frame clip | **Yes** | Slow | 0.87–0.92 | Videos with motion artifacts |

---

## 6. Backend (Inference API)

**File:** `inference_api.py`  
**Framework:** FastAPI (Python)  
**Port:** 8000

### Responsibilities
- Load all trained models at startup (auto-discovers latest checkpoints)
- Accept image/video uploads via HTTP
- Preprocess files into tensors
- Run inference on the selected model
- Return structured JSON result

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Server status + which models are loaded |
| `GET` | `/health` | Liveness check |
| `GET` | `/models` | List models with checkpoint paths |
| `POST` | `/predict?model=xception` | Upload file → get prediction |

### Model Loading Strategy
At startup, for each model key (`xception`, `ucf`, `efficientnetb4`):
1. Scans `logs/training/` for all folders starting with `<model>_`
2. Sorts by timestamp (newest first)
3. Loads the first `test/avg/ckpt_best.pth` found
4. Model is marked unavailable if no checkpoint exists yet

```python
# Auto-discovery example
logs/training/
├── xception_2026-03-01-21-45-05/test/avg/ckpt_best.pth  ← loaded
├── xception_2026-03-01-21-44-22/                        ← skipped (older)
└── efficientnetb4_2026-03-01-17-42-51/test/avg/ckpt_best.pth  ← loaded
```

### Image Preprocessing Pipeline
```
PIL Image
    → convert RGB
    → Resize to 256×256
    → ToTensor (HWC→CHW, [0,255]→[0,1])
    → Normalize (mean=0.5, std=0.5 → range [-1, 1])
    → unsqueeze(0) → shape [1, 3, 256, 256]
    → .to(DEVICE)  # GPU if available
```

### Video Processing Pipeline
```
Raw video bytes
    → write to temp .mp4 file
    → OpenCV VideoCapture
    → sample 16 evenly-spaced frame indices
    → decode each frame (BGR→RGB)
    → apply same image pipeline to each
    → torch.cat → shape [16, 3, 256, 256]
    → average softmax scores across all 16 frames
    → delete temp file
```

---

## 7. Frontend (Web UI)

**Framework:** React 18 + Vite  
**Port:** 3000  
**Styling:** Tailwind CSS  
**Location:** `frontend/src/`

### Component Architecture

```
App.jsx  (state management, API calls)
│
├── SolarSystemBackground.jsx   (Three.js 3D background, WebGL canvas)
├── Header.jsx                  (Logo, title, animated border)
├── ElectricBorder.jsx          (Animated electric border effect)
│
├── UploadZone.jsx              ← Main interaction area
│   ├── react-dropzone          (drag & drop file handling)
│   ├── Model selector          (Xception / UCF / EfficientNetB4 pills)
│   ├── Image/video preview     (inline with remove button)
│   └── Analyze button          (triggers API call)
│
├── ResultCard.jsx              ← Result display
│   ├── Verdict banner          (REAL / FAKE with color coding)
│   ├── Confidence bar          (animated fill)
│   ├── Probability breakdown   (Real % vs Fake % cards)
│   ├── File metadata           (type, frames analyzed, model used)
│   └── Reset button
│
├── HowItWorks.jsx              (Explainer cards section)
└── Footer.jsx
```

### State Flow (App.jsx)
```
useState:
  file        → the raw File object
  preview     → { url, type: 'image'|'video' }
  result      → API response JSON
  loading     → boolean (spinner state)
  error       → error message string
  model       → 'xception' | 'ucf' | 'efficientnetb4'

User drops file
    → handleFile() → setFile + setPreview

User clicks Analyze
    → handleAnalyze()
    → FormData POST to /predict?model={model}
    → setResult(response.data)

User clicks Reset
    → handleReset() → clear all state
```

### API Communication
```javascript
// In App.jsx
axios.post(`/predict?model=${model}`, formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
  timeout: 120_000   // 2 min timeout for large videos
})
```

The `VITE_API_URL` env variable controls the base URL.  
Default (empty) means same-origin → proxied by Vite to `localhost:8000`.

---

## 8. Storage & Data Layer

DeepGuard uses **no traditional database**. Data is stored as files on disk:

| Data type | Location | Format | Size |
|---|---|---|---|
| Raw videos | *(user-provided)* | MP4 | ~200 GB |
| Extracted face frames | `datasets/rgb/` | PNG 256×256 | ~15–30 GB |
| Dataset split index | `preprocessing/dataset_json/DFD.json` | JSON | ~2 MB |
| Model weights (pretrained) | `training/pretrained/` | `.pth` (PyTorch) | ~500 MB |
| Trained checkpoints | `logs/training/*/test/avg/ckpt_best.pth` | `.pth` | ~100–300 MB each |
| Training metrics | `logs/training/*/train/*/metric_board/` | TensorBoard events | ~5–50 MB |
| Training logs | `logs/training/*/training.log` | Plain text | ~1–5 MB |

### Why no database?
- All training data is **static** — loaded once and never changed
- The `DFD.json` file acts as a dataset index (replaces a database query)
- Inference is stateless — no user sessions, uploads are not persisted
- Checkpoints are binary files best stored on disk, not in a DB

---

## 9. Complete Tech Stack

### AI / ML
| Technology | Purpose |
|---|---|
| **PyTorch** | Deep learning framework (training + inference) |
| **CUDA** | GPU acceleration (NVIDIA) |
| **torchvision** | Image transforms, pretrained model loading |
| **OpenCV** | Video frame extraction, face detection (Haar cascade) |
| **Pillow (PIL)** | Image loading and format handling |
| **albumentations** | Training data augmentation |
| **scikit-learn** | AUC/metric computation |
| **timm** | Pretrained model zoo (EfficientNet, ViT etc.) |
| **einops** | Tensor reshape operations (used by video models) |
| **TensorBoard** | Training metric visualization |

### Backend
| Technology | Purpose |
|---|---|
| **Python 3.10** | Language |
| **FastAPI** | REST API web framework |
| **Uvicorn** | ASGI server (runs FastAPI) |
| **PyYAML** | Load model config YAML files |
| **python-multipart** | Parse file upload form data |

### Frontend
| Technology | Purpose |
|---|---|
| **React 18** | UI component framework |
| **Vite 5** | Build tool and dev server |
| **Tailwind CSS** | Utility-first CSS styling |
| **Three.js** | 3D graphics (WebGL) |
| **@react-three/fiber** | React renderer for Three.js |
| **@react-three/drei** | Three.js helper components (orbit controls, stars etc.) |
| **Framer Motion** | Animations and transitions |
| **axios** | HTTP requests to the API |
| **react-dropzone** | Drag & drop file upload |
| **lucide-react** | Icon library |

### Infrastructure
| Technology | Purpose |
|---|---|
| **PowerShell** | Launch scripts (`start.ps1`, `run_training_sequence.ps1`) |
| **NVIDIA GPU** | Required for practical training speed |
| **Python venv** | Isolated Python environment (`.venv/`) |
| **npm** | Frontend package manager |

---

## 10. How a Prediction is Made (Step-by-Step)

Here is the complete journey of a single image upload through the system:

```
Step 1  USER
        Opens browser → http://localhost:5173
        Selects "Xception" model (or default)
        Drags an image onto the upload zone

Step 2  REACT FRONTEND (UploadZone.jsx)
        react-dropzone captures the File object
        Creates a local object URL for preview
        Shows the image preview in the UI
        User clicks "Analyze Now"

Step 3  REACT FRONTEND (App.jsx)
        Sets loading=true (shows spinner)
        Wraps file in FormData
        POST http://localhost:8000/predict?model=xception
        Content-Type: multipart/form-data

Step 4  FASTAPI SERVER (inference_api.py)
        Receives the uploaded bytes
        Detects file type from content-type header
        Calls: img = Image.open(io.BytesIO(data))
        Applies preprocessing:
            → Resize(256, 256)
            → ToTensor() → shape [3, 256, 256]
            → Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            → unsqueeze(0) → shape [1, 3, 256, 256]
            → .to("cuda")

Step 5  XCEPTION MODEL (GPU)
        data_dict = {"image": tensor, "label": zeros}
        output = model(data_dict, inference=True)
        Returns dict with key "cls" → logits [1, 2]
        F.softmax(logits, dim=1) → [[0.05, 0.95]]
        real_prob = 0.05 (5%)
        fake_prob = 0.95 (95%)

Step 6  FASTAPI SERVER
        label      = "fake"   (fake_prob > real_prob)
        confidence = 95.0%
        Returns JSON:
        {
          "label": "fake",
          "confidence": 95.0,
          "prob_real": 5.0,
          "prob_fake": 95.0,
          "file_type": "image",
          "filename": "face.jpg",
          "model": "xception"
        }

Step 7  REACT FRONTEND (App.jsx)
        setResult(data)      → triggers ResultCard render
        setLoading(false)    → hides spinner

Step 8  REACT FRONTEND (ResultCard.jsx)
        Shows large "FAKE" banner (red)
        Animates confidence bar to 95%
        Shows probability cards (5% Real, 95% Fake)
        Shows "Xception" model badge
        Shows "Image" file type badge
```

**Total latency (GPU):** ~50–200ms for images, ~500ms–2s for videos
