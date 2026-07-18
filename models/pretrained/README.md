# Pretrained Models

This directory contains pretrained model weights for the tutorial notebooks.

## Available Models

| Model | File | Dataset | Classes | Size | Training |
|-------|------|---------|---------|------|----------|
| Brain Tumor Segmentation | `brain_tumor_unet3d.pt` | Decathlon Task01 | 4 | ~8 MB | 100 epochs |

## Model Details

### brain_tumor_unet3d.pt

- **Architecture**: 3D U-Net (depth=4, base_features=32)
- **Input**: 4-channel MRI (T1, T1ce, T2, FLAIR), shape (4, 128, 128, 128)
- **Output**: 4-class segmentation (background, necrotic, edema, enhancing)
- **Training**: Up to 100 epochs on Medical Segmentation Decathlon Task01_BrainTumour (with early stopping)
- **Metric**: Whole-tumor Dice score (tumor vs background overlap)

## Usage

```python
import torch
from src.models import UNet3D

# Load pretrained model
model = UNet3D(in_channels=4, out_channels=4, base_features=32, depth=4)
checkpoint = torch.load('models/pretrained/brain_tumor_unet3d.pt', map_location='cpu')
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
tumor region overlap. State-of-the-art BraTS models typically achieve 0.80-0.90.

If using an older pretrained model, retrain with `USE_PRETRAINED = False`.

## File Size Policy

Only small models (<10 MB) are tracked in Git. Larger models should be:
- Hosted on Hugging Face Hub, Zenodo, or similar
- Downloaded on-demand via the notebook
