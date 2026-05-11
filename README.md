# DeepGuard

An AI-powered deepfake detection system that identifies fake images and videos using deep learning.

## What is DeepGuard?

DeepGuard detects whether an uploaded image or video contains a deepfake. Users can upload media through a web interface and receive a verdict (**Real** or **Fake**) with confidence percentage.

The system is trained on the **DFD (Deep Fake Detection)** dataset and uses multiple detection models including Xception, EfficientNetB4, I3D, and UCF.

## Key Features

- **Web Interface**: React-based UI for easy media upload and analysis
- **Multiple Models**: Xception, EfficientNetB4, I3D, UCF detectors
- **FastAPI Backend**: REST API for predictions
- **GPU Optimized**: CUDA support for fast inference
- **Real-time Processing**: ~50–200ms for images, ~500ms–2s for videos (GPU)

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Inference API
```bash
python inference_api.py
```
API runs on `http://localhost:8000`

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
Web interface runs on `http://localhost:5173`

## Project Structure

```
├── training/           # Model training pipeline
├── inference_api.py    # FastAPI server for predictions
├── frontend/           # React web UI (Vite + Tailwind)
├── datasets/           # Training datasets
├── preprocessing/      # Data preparation scripts
└── logs/               # Training checkpoints
```

## Tech Stack

- **Backend**: FastAPI, Python, PyTorch
- **Frontend**: React, Vite, Tailwind CSS
- **Models**: Xception, EfficientNetB4, I3D, UCF
- **Dataset**: Deep Fake Detection (DFD)


