# Pretrained Models

This directory contains pretrained model weights for the tutorial notebooks.

## Available Models

| Model | File | Dataset | Arch | Size | Notes |
|-------|------|---------|------|------|-------|
| Brain Tumor Segmentation (default) | `brain_tumor_unet3d.pt` | Decathlon Task01 (100-case subset) | depth=4, base=32 | ~87 MB | Use this in Notebook 01 |
| A5000 training run | `brain_tumor_unet3d_A5000.pt` | same | depth=4, base=32 | ~87 MB | Same architecture/metrics as default |
| Older macOS run | `brain_tumor_unet3d_macos_20260123.pt` | same | depth=3, base=16 | ~5 MB | **Do not use** — inflated Dice from pre-fix metric |

The default weights are also published on the [`v1.1-data`](https://github.com/arvidl/medAI-hands-on/releases/tag/v1.1-data) GitHub Release.

## Model Details

### `brain_tumor_unet3d.pt` (default)

- **Architecture**: 3D U-Net (`depth=4`, `base_features=32`)
- **Input**: 4-channel MRI in **Decathlon channel order** `(FLAIR, T1, T1ce, T2)`, shape `(4, 128, 128, 128)`
- **Output**: 4-class segmentation (background, necrotic, edema, enhancing)
- **Training**: Up to 100 epochs on the 100-case subset (early stopping; checkpoint epoch ≈ 71)
- **Metric**: Whole-tumor Dice ≈ **0.82** (tumor vs background overlap on validation)

## Usage

```python
import torch
from src.models import UNet3D

# Load pretrained model
model = UNet3D(in_channels=4, out_channels=4, base_features=32, depth=4)
checkpoint = torch.load('models/pretrained/brain_tumor_unet3d.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Check training info
print(f"Trained for {checkpoint['epoch']} epochs")
print(f"Best validation Dice: {checkpoint['best_dice']:.4f}")
```

## Retraining

To retrain the model from scratch, set `USE_PRETRAINED = False` in the notebook configuration section.

### Important: Metric Fix (2026-02-01)

The `compute_dice_score` function was corrected to compute **whole-tumor Dice**
(measuring overlap between predicted and ground truth tumor regions) instead of
incorrectly counting all matching voxels including background.

**Previous behavior (buggy)**: Reported ~0.96 Dice because it counted background
matches as "correct", despite actual tumor overlap being only ~0.13.

**Current behavior (correct)**: Reports realistic Dice scores based on actual
tumor region overlap. State-of-the-art BraTS models typically achieve 0.80–0.90.

If using an older pretrained model (e.g. `brain_tumor_unet3d_macos_20260123.pt`),
prefer the default checkpoint or retrain with `USE_PRETRAINED = False`.

## File Size Policy

The default U-Net checkpoint (~87 MB) is tracked in git under `models/pretrained/`
(via an allow-list in `.gitignore`) and mirrored on the `v1.1-data` release.
Larger assets (full Decathlon, MedSAM2 weights) remain download-on-demand and are
not stored in the repository.
