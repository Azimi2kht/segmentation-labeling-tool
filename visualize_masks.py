import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


IMAGE_EXTENSIONS = (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass
class SamplePaths:
    mask_path: str
    image_path: Optional[str]
    band_index: Optional[int]


def collect_mask_files(output_dir: str, mask_path: Optional[str] = None) -> List[str]:
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


def parse_mask_stem_and_band(mask_path: str) -> tuple[str, Optional[int]]:
    mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
    match = re.match(r"^(?P<base>.+)_band(?P<band>\d+)$", mask_stem)
    if match:
        return match.group("base"), int(match.group("band"))
    return mask_stem, None


def resolve_image_path(mask_path: str, images_dir: Optional[str]) -> Optional[str]:
    if images_dir is None:
        return None

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    stem, _ = parse_mask_stem_and_band(mask_path)
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_dir, f"{stem}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mn = float(np.min(image))
    mx = float(np.max(image))
    return (image - mn) / (mx - mn + 1e-8)


def to_rgb_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        g = normalize_grayscale(image)
        return np.stack([g, g, g], axis=-1)
    if image.ndim == 3:
        if image.shape[2] == 3:
            return normalize_grayscale(image)
        if image.shape[2] == 4:
            return normalize_grayscale(image[:, :, :3])
        # Generic HSI/hypercube fallback for visualization.
        mean_img = np.mean(image, axis=2)
        g = normalize_grayscale(mean_img)
        return np.stack([g, g, g], axis=-1)
    raise ValueError(f"Unsupported image shape for display: {image.shape}")


def load_original_image(
    image_path: Optional[str],
    expected_shape: tuple[int, int],
    band_index: Optional[int],
) -> np.ndarray:
    if image_path is None:
        h, w = expected_shape
        return np.zeros((h, w, 3), dtype=np.float32)

    if image_path.endswith(".npy"):
        raw = np.load(image_path)
    else:
        raw = plt.imread(image_path)
    if raw.ndim == 3 and raw.shape[2] > 4:
        if band_index is None:
            band_index = 0
        if band_index < 0 or band_index >= raw.shape[2]:
            raise ValueError(
                f"Band index {band_index} out of range for image {image_path} with {raw.shape[2]} bands"
            )
        raw = raw[:, :, band_index]

    rgb = to_rgb_display(raw)
    if rgb.shape[:2] != expected_shape:
        raise ValueError(
            f"Image shape {rgb.shape[:2]} does not match mask shape {expected_shape} for {image_path}"
        )
    return rgb


class MaskBrowser:
    def __init__(self, samples: List[SamplePaths]):
        if not samples:
            raise ValueError("No samples to visualize.")
        self.samples = samples
        self.idx = 0

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 6))
        self.fig.subplots_adjust(bottom=0.18, wspace=0.06)

        self.prev_button = Button(
            self.fig.add_axes([0.34, 0.05, 0.12, 0.06]), "Prev [A/Left]"
        )
        self.next_button = Button(
            self.fig.add_axes([0.54, 0.05, 0.12, 0.06]), "Next [D/Right]"
        )
        self.prev_button.on_clicked(lambda _event: self.move(-1))
        self.next_button.on_clicked(lambda _event: self.move(+1))

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.render()

    def on_key_press(self, event) -> None:
        if event.key in ("left", "a"):
            self.move(-1)
        elif event.key in ("right", "d"):
            self.move(+1)
        elif event.key == "q":
            plt.close(self.fig)

    def move(self, delta: int) -> None:
        self.idx = (self.idx + delta) % len(self.samples)
        self.render()

    def render(self) -> None:
        sample = self.samples[self.idx]
        mask = np.load(sample.mask_path)
        if mask.ndim != 2:
            raise ValueError(
                f"Expected 2D mask, got shape {mask.shape} in {sample.mask_path}"
            )

        unique_labels = np.unique(mask)
        base = load_original_image(sample.image_path, mask.shape, sample.band_index)
        overlay_alpha = np.where(mask > 0, 0.45, 0.0)

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        self.axes[0].imshow(base, interpolation="nearest")
        self.axes[0].set_title("Original image")

        self.axes[1].imshow(mask, cmap="tab20", interpolation="nearest")
        self.axes[1].set_title("Mask")

        self.axes[2].imshow(base, interpolation="nearest")
        self.axes[2].imshow(
            mask, cmap="tab20", interpolation="nearest", alpha=overlay_alpha
        )
        self.axes[2].set_title("Overlay")

        image_name = (
            os.path.basename(sample.image_path) if sample.image_path else "not found"
        )
        band_text = (
            f"band={sample.band_index}" if sample.band_index is not None else "band=n/a"
        )
        fig_title = (
            f"[{self.idx + 1}/{len(self.samples)}] mask={os.path.basename(sample.mask_path)} | "
            f"image={image_name} | {band_text} | shape={mask.shape} | labels={unique_labels[:15]}"
            + ("..." if len(unique_labels) > 15 else "")
        )
        self.fig.suptitle(fig_title, fontsize=10)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize segmentation masks with original image and overlay. "
            "Navigation: A/Left=prev, D/Right=next, Q=quit."
        )
    )
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
        "--images_dir",
        type=str,
        default="./data",
        help=(
            "Directory containing original images with the same stem as mask files. "
            "Supports .npy and common image formats."
        ),
    )
    parser.add_argument(
        "--latest_only",
        action="store_true",
        help="If set, only show the latest mask file by filename order",
    )
    args = parser.parse_args()

    mask_files = collect_mask_files(args.output_dir, args.mask_path)
    if args.latest_only and args.mask_path is None:
        mask_files = [mask_files[-1]]

    samples = []
    for p in mask_files:
        _, band_index = parse_mask_stem_and_band(p)
        samples.append(
            SamplePaths(
                mask_path=p,
                image_path=resolve_image_path(p, args.images_dir),
                band_index=band_index,
            )
        )
    missing_images = [
        s.mask_path
        for s in samples
        if s.image_path is None and args.images_dir is not None
    ]
    if missing_images:
        print(
            f"Warning: missing originals for {len(missing_images)} mask(s) in {args.images_dir}."
        )

    print(f"Visualizing {len(samples)} sample(s).")
    browser = MaskBrowser(samples)
    browser.show()


if __name__ == "__main__":
    main()
