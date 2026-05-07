import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def collect_mask_files(output_dir: str, mask_path: str = None) -> List[str]:
    if mask_path:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        return [mask_path]

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    files = [
        os.path.join(output_dir, f)
        for f in sorted(os.listdir(output_dir))
        if f.endswith(".npy")
    ]
    if not files:
        raise FileNotFoundError(f"No .npy mask files found in: {output_dir}")
    return files


def show_mask(mask_path: str):
    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape} in {mask_path}")

    unique_labels = np.unique(mask)
    plt.figure(figsize=(7, 6))
    plt.imshow(mask, cmap="tab20", interpolation="nearest")
    plt.title(
        f"{os.path.basename(mask_path)}\n"
        f"shape={mask.shape}, labels={unique_labels[:15]}"
        + ("..." if len(unique_labels) > 15 else "")
    )
    plt.colorbar(label="Class ID")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize saved segmentation masks")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory containing saved mask .npy files",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Optional single mask file path to visualize",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If set, visualize all masks in output_dir one-by-one",
    )
    args = parser.parse_args()

    mask_files = collect_mask_files(args.output_dir, args.mask_path)
    if not args.all and args.mask_path is None:
        mask_files = [mask_files[-1]]  # default: latest by filename order

    print(f"Visualizing {len(mask_files)} mask file(s).")
    for mask_file in mask_files:
        print(f"- {mask_file}")
        show_mask(mask_file)


if __name__ == "__main__":
    main()
