import os
import json
import random
from pathlib import Path

                                                                                
BASE = Path(r"F:\repo's\DeepfakeBench-main\DeepfakeBench-main\datasets\rgb\Celeb-DF-v2")
FAKE_FRAMES = BASE / "fake" / "frames"
REAL_FRAMES = BASE / "real" / "frames"

OUTPUT_JSON = Path(r"F:\repo's\DeepfakeBench-main\DeepfakeBench-main\preprocessing\dataset_json\DFD.json")

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
                                

SEED = 42
                                                                                

random.seed(SEED)


def collect_split(frames_dir: Path, label: str) -> dict:
\
\
\
       
    if not frames_dir.exists():
        print(f"[WARN] Frames directory not found: {frames_dir}")
        print("       Run  python preprocessing/preprocess_dfd.py  first.")
        return {"train": {}, "val": {}, "test": {}}

    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    if not video_dirs:
        print(f"[WARN] No video frame folders found in {frames_dir}")
        return {"train": {}, "val": {}, "test": {}}

                        
    video_dirs = [d for d in video_dirs if any(d.glob("*.png"))]
    print(f"  {label}: {len(video_dirs)} videos with frames")

                     
    random.shuffle(video_dirs)
    n = len(video_dirs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": video_dirs[:n_train],
        "val":   video_dirs[n_train:n_train + n_val],
        "test":  video_dirs[n_train + n_val:],
    }

    result = {"train": {}, "val": {}, "test": {}}
    for split_name, dirs in splits.items():
        for vdir in dirs:
            frames = sorted(str(p) for p in vdir.glob("*.png"))
            if frames:
                result[split_name][vdir.name] = {"label": label, "frames": frames}

    for split_name, d in result.items():
        print(f"    {split_name}: {len(d)} videos")

    return result


def main():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Collecting fake frames...")
    fake_split = collect_split(FAKE_FRAMES, "DFD_fake")

    print("Collecting real frames...")
    real_split = collect_split(REAL_FRAMES, "DFD_real")

    dataset_json = {
        "DFD": {
            "DFD_fake": fake_split,
            "DFD_real": real_split,
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset_json, f)

    total_fake_train = len(fake_split["train"])
    total_real_train = len(real_split["train"])
    print(f"\n✓ Saved: {OUTPUT_JSON}")
    print(f"  Train  — fake: {total_fake_train}, real: {total_real_train}")
    print(f"  Val    — fake: {len(fake_split['val'])},  real: {len(real_split['val'])}")
    print(f"  Test   — fake: {len(fake_split['test'])},  real: {len(real_split['test'])}")
    print("\nNext step: start training with")
    print("  python training/train.py --detector_path training/config/detector/efficientnetb4_dfd.yaml")


if __name__ == "__main__":
    main()
