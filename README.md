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
python manual_segmentation.py --npy_path "./exp6_003_preprocessed_cropped_rotated/frame_0160_radiance_cropped_rotated.npy"
```

## 3) Save output

- Click `Save` (or press `s`)
- Masks are saved in `output/`
- Filename includes the current band, for example:
  - `output/frame_0160_radiance_cropped_rotated_band091.npy`

## 4) Verify saved masks

Show latest mask:

```bash
python visualize_masks.py
```

Show one specific file:

```bash
python visualize_masks.py --mask_path "./output/frame_0160_radiance_cropped_rotated_band091.npy"
```

Show all masks in `output/`:

```bash
python visualize_masks.py --all
```
