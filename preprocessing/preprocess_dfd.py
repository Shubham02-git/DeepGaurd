"""
Preprocessing script for DFD (DeepFakeDetection) dataset stored as:
  datasets/rgb/Celeb-DF-v2/fake/DFD_manipulated_sequences/*.mp4
  datasets/rgb/Celeb-DF-v2/real/DFD_original sequences/*.mp4

Extracts face crops from videos using OpenCV Haar cascade detector,
saves them as PNG frames under a 'frames/' sibling directory.

Output structure:
  datasets/rgb/Celeb-DF-v2/fake/frames/<video_stem>/<000.png 001.png ...>
  datasets/rgb/Celeb-DF-v2/real/frames/<video_stem>/<000.png 001.png ...>
"""

import os
import cv2
import sys
import time
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path(r"F:\repo's\DeepfakeBench-main\DeepfakeBench-main\datasets\rgb\Celeb-DF-v2")
FAKE_DIR = BASE / "fake" / "DFD_manipulated_sequences"
REAL_DIR = BASE / "real" / "DFD_original sequences"
FAKE_FRAMES = BASE / "fake" / "frames"
REAL_FRAMES = BASE / "real" / "frames"

NUM_FRAMES   = 32          # frames to extract per video
RESOLUTION   = 256         # output face crop size
SCALE_FACTOR = 1.3         # face crop scale (adds padding around detected box)
MAX_WORKERS  = os.cpu_count() or 4
LOG_PATH     = Path(__file__).parent / "logs" / "preprocess_dfd.log"
# ──────────────────────────────────────────────────────────────────────────────

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def get_face_cascade():
    cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    assert not cc.empty(), "Failed to load Haar cascade"
    return cc


def crop_face(frame: np.ndarray, cascade, resolution: int, scale: float) -> np.ndarray | None:
    """Detect the largest face in *frame* and return a square crop at *resolution*."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # pick the largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    # add padding
    pad = int(max(w, h) * (scale - 1) / 2)
    H, W = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (resolution, resolution), interpolation=cv2.INTER_AREA)


def extract_video(video_path: Path, out_dir: Path, num_frames: int, resolution: int, scale: float) -> int:
    """Extract *num_frames* face-cropped PNG files from *video_path* into *out_dir*."""
    cascade = get_face_cascade()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        log.warning(f"No frames in {video_path}")
        return 0

    # evenly-spaced frame indices
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    vid_out_dir = out_dir / video_path.stem
    vid_out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, idx in enumerate(indices):
        img_path = vid_out_dir / f"{i:03d}.png"
        if img_path.exists():          # skip already extracted
            saved += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        face = crop_face(frame, cascade, resolution, scale)
        if face is None:
            # fallback: centre-crop without face detection
            h, w = frame.shape[:2]
            side = min(h, w)
            top  = (h - side) // 2
            left = (w - side) // 2
            face = cv2.resize(frame[top:top+side, left:left+side], (resolution, resolution))

        cv2.imwrite(str(img_path), face)
        saved += 1

    cap.release()
    return saved


def process_folder(video_dir: Path, frames_dir: Path, label: str) -> None:
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        log.error(f"No .mp4 files found in {video_dir}")
        return

    log.info(f"Processing {len(videos)} {label} videos -> {frames_dir}")
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(extract_video, v, frames_dir, NUM_FRAMES, RESOLUTION, SCALE_FACTOR): v
            for v in videos
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=label):
            v = futures[fut]
            try:
                n = fut.result()
                if n == 0:
                    log.warning(f"0 frames saved for {v.name}")
            except Exception as e:
                log.error(f"Failed {v.name}: {e}")

    elapsed = (time.monotonic() - t0) / 60
    log.info(f"{label}: done in {elapsed:.1f} min")


if __name__ == "__main__":
    log.info("=== DFD Preprocessing Start ===")
    process_folder(FAKE_DIR, FAKE_FRAMES, "fake")
    process_folder(REAL_DIR, REAL_FRAMES, "real")
    log.info("=== DFD Preprocessing Complete ===")
    log.info(f"Fake frames: {FAKE_FRAMES}")
    log.info(f"Real frames: {REAL_FRAMES}")
    log.info("Next step: run  python preprocessing/generate_dfd_json.py")
