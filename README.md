# Manual HSI Segmentation Tool

Simple interactive labeling tool for hyperspectral `.npy` cubes (`H, W, B`).

## What this project includes

- `manual_segmentation.py`: create segmentation masks band-by-band
- `visualize_masks.py`: quickly check saved mask files

## 1) Install

Use your existing conda env (`h2`) or any Python 3 env.

```bash
pip install -r requirements.txt
```

## 2) Run the segmentation tool

```bash
python manual_segmentation.py --npy_path "./data/frame_0160_radiance_cropped_rotated.npy"
```

## 3) How saving works (important)

- Click `Save` (or press `s`)
- The tool saves one 2D mask (`H, W`) for the **currently selected band**
- Output name is built from `--out_mask` (or default `./output/<input_filename>.npy`) and then appends the current band:
  - `<root>_band<BBB>.npy` where `BBB` is zero-padded (e.g. `017`, `091`)
- Example:
  - `output/frame_0160_radiance_cropped_rotated_band091.npy`
- This means if you switch to another band and press save again, it creates another file with that new band suffix.

## 4) Verify saved masks

The mask viewer shows three panels:

- Original image band used by the mask
- Mask
- Mask overlay on the original band

It also supports navigation (`Prev/Next` buttons, `A/Left`, `D/Right`, `Q` to quit).

By default, it uses:

- masks from `./output`
- original cubes from `./data`
- all masks (so navigation works immediately)

Show all masks:

```bash
python visualize_masks.py
```

Show latest mask only:

```bash
python visualize_masks.py --latest_only
```

Show one specific mask file:

```bash
python visualize_masks.py --mask_path "./output/frame_0160_radiance_cropped_rotated_band091.npy"
```

Use a custom originals directory:

```bash
python visualize_masks.py --images_dir "/path/to/original/cubes"
```
