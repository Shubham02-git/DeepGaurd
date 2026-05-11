import argparse
import copy
import json
import random
from pathlib import Path


def balance_split(label_sections, split, seed):
    rng = random.Random(seed)
    real_items = list(label_sections["DFD_real"][split].items())
    fake_items = list(label_sections["DFD_fake"][split].items())
    target = min(len(real_items), len(fake_items))

    balanced_real = dict(rng.sample(real_items, target))
    balanced_fake = dict(rng.sample(fake_items, target))

    label_sections["DFD_real"][split] = balanced_real
    label_sections["DFD_fake"][split] = balanced_fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="preprocessing/dataset_json/DFD.json")
    parser.add_argument("--output-dir", default="preprocessing/dataset_json_balanced")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "DFD.json"

    data = json.loads(input_path.read_text())
    balanced = copy.deepcopy(data)
    label_sections = balanced["DFD"]

    for split in ("train", "test"):
        balance_split(label_sections, split, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(balanced))

    for split in ("train", "test"):
        real_count = len(label_sections["DFD_real"][split])
        fake_count = len(label_sections["DFD_fake"][split])
        print(f"{split}: real={real_count}, fake={fake_count}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
