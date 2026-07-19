# Data Directory

This directory stores datasets used in the notebooks. 

## ⚠️ Git Tracking Policy

| Dataset | Size | Tracked in Git? | Notes |
|---------|------|-----------------|-------|
| `README.md` | 5 KB | ✅ Yes | Documentation only |
| `brats_subset_100.zip` | ~900 MB | 📦 **GitHub Release** (`v1.1-data`) | Subset for Colab/local (100 cases) |
| `brats_sample/` | ~21 MB | ❌ No (generated) | Small synthetic data for quick demos |
| `decathlon_brain/` | **~7.5 GB** download / **~16 GB** extracted | ❌ **No** | Downloaded on-demand via notebook |
| `BraTS*/` | Variable | ❌ No | Requires manual registration & download |

**Large datasets are excluded via `.gitignore`** - they are downloaded automatically when running the notebooks.

---

## Quick Start

### Option 1: GitHub Subset (Recommended for Colab) ⭐

The **100-case BRATS subset** is hosted as a GitHub Release for fast, reliable access:

```python
# Automatic in Colab, or force with:
DATA_SOURCE = "github_subset"  # Downloads ~900 MB from GitHub Release
```

| Property | Value |
|----------|-------|
| Cases | BRATS_001 through BRATS_100 (100 cases) |
| Size | ~900 MB download |
| Format | Same as Decathlon (NIfTI, 4 modalities + labels) |
| Splits | 70 train / 15 validation / 15 test (SEED=42) |

### Option 2: Full Decathlon (Local Development)

Download the complete Medical Segmentation Decathlon dataset:

```python
DATA_SOURCE = "decathlon_full"  # Downloads ~7.5 GB, 484 cases
```

### Option 3: Synthetic Data (Code Testing)

Auto-generated brain-like volumes for testing without real data:

```python
data_dir = download_sample_data(dataset="brats_sample")  # Instant, ~5 samples
```

---

## Dataset Options

### 1. Medical Segmentation Decathlon - Brain Tumor (full dataset)

**Best for**: Local development with the complete cohort (484 training cases). For Colab or a quick start, prefer the GitHub subset above (`github_subset`, ~900 MB).

| Property | Value |
|----------|-------|
| Task | Task01_BrainTumour |
| Samples | 484 training, 266 test |
| Modalities | FLAIR, T1w, T1-Gd (contrast), T2w |
| Segmentation | 3 classes: edema, enhancing, necrotic |
| Size | 7.5 GB download, ~16 GB extracted |
| Format | NIfTI (.nii.gz) |

**Automatic download** (in notebook):
```python
from src.data_utils import download_sample_data
data_dir = download_sample_data(data_dir="data", dataset="decathlon")
```

**Manual download**:
1. Go to http://medicaldecathlon.com/
2. Download `Task01_BrainTumour.tar` (7.5 GB)
3. Extract to `data/decathlon_brain/`

### 2. BraTS Challenge Dataset (Gold Standard)

**Best for**: Research publications, benchmarking against literature

| Property | Value |
|----------|-------|
| Samples | ~2,000+ across years |
| Modalities | T1, T1-Gd, T2, FLAIR |
| Segmentation | Enhancing tumor, tumor core, whole tumor |
| Access | Requires registration |

**Download instructions**:
1. Create account at [Synapse](https://www.synapse.org/)
2. Go to [BraTS 2023](https://www.synapse.org/#!Synapse:syn51156910) (or latest year)
3. Request access and agree to data use terms
4. Download and extract to `data/BraTS2023/`
5. Update notebook paths accordingly

**BraTS Citation** (if using in publications):
```bibtex
@article{menze2015brats,
  title={The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)},
  author={Menze, Bjoern H and others},
  journal={IEEE TMI},
  year={2015}
}
```

### 3. Synthetic Data (Demo Only)

Auto-generated brain-like volumes for testing code without real data.

```python
data_dir = download_sample_data(dataset="brats_sample")  # 5 synthetic samples
```

---

## Google Colab Usage

### Default Behavior (Recommended) ⭐

Notebook 01 first clones the repository (for `src/` imports), then downloads the **100-case subset** from GitHub Releases when `DATA_SOURCE = "github_subset"`:

```python
# Automatic in Colab - downloads ~900 MB in a few minutes
DATA_SOURCE = "github_subset"  # This is the default in Colab
```

This provides:
- Fast download (~900 MB vs 7.5 GB for the full Decathlon task)
- Real brain tumor MRI data (BRATS_001–BRATS_100)
- Same reproducible train/val/test splits (70 / 15 / 15, SEED=42)

### Full Dataset in Colab

To use the complete Decathlon dataset in Colab:

```python
# Override the automatic selection
DATA_SOURCE = "decathlon_full"  # Downloads ~7.5 GB, takes 10-30 min
```

### Persistent Storage in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

# Download once to Drive (persists across sessions)
# Modify base_data_dir in the notebook:
base_data_dir = "/content/drive/MyDrive/medAI_data"
```

---

## Directory Structure After Download

```
data/
├── README.md                           # This file
│
├── brats_subset/                       # GitHub Release subset (Colab default)
│   └── Task01_BrainTumour/
│       ├── imagesTr/                   # 100 images (BRATS_001–BRATS_100)
│       │   ├── BRATS_001.nii.gz        # 4-channel: FLAIR, T1, T1ce, T2
│       │   └── ...
│       └── labelsTr/                   # 100 labels
│           ├── BRATS_001.nii.gz        # Segmentation mask
│           └── ...
│
├── decathlon_brain/                    # Full Decathlon (local default)
│   └── Task01_BrainTumour/
│       ├── dataset.json
│       ├── imagesTr/                   # Training images (484 cases)
│       ├── labelsTr/                   # Training labels
│       └── imagesTs/                   # Test images (no labels)
│
├── brats_sample/                       # Synthetic data (auto-generated)
│   ├── sample_000/
│   │   ├── t1.npy, t1ce.npy, t2.npy, flair.npy, seg.npy
│   └── sample_001/ ...
│
└── BraTS2023/                          # Official BraTS (manual download)
    └── ... (follows BraTS structure)
```

---

## GitHub Release Data

The 100-case BRATS subset is hosted as a GitHub Release asset for reliable access from Colab.

### For Maintainers: Updating the Release

If you need to update the data subset:

1. Create the subset zip (from a full Decathlon download):
```bash
python scripts/create_brats_subset.py \
  --input data/decathlon_brain/Task01_BrainTumour \
  --output brats_subset_100.zip \
  --num-cases 100
```

2. Create a GitHub Release:
   - Go to: https://github.com/arvidl/medAI-hands-on/releases/new
   - Tag: `v1.1-data`
   - Title: "BRATS Subset Data (100 cases)"
   - Description: "100-case subset of Medical Segmentation Decathlon brain tumor data for Colab/local notebooks"
   - Attach: `brats_subset_100.zip` (optional: `brain_tumor_unet3d.pt`)
   - Publish

3. The download URL will be:
```
https://github.com/arvidl/medAI-hands-on/releases/download/v1.1-data/brats_subset_100.zip
```

---

## Data Privacy Notice

When working with medical imaging data:

- **Never commit real patient data** to public repositories
- **Follow institutional IRB guidelines** for data handling
- **Use only de-identified data** from approved sources
- **Cite data sources** appropriately in publications

The Decathlon and BraTS datasets are fully de-identified and approved for research use.

---

## Troubleshooting

### Download fails in Colab
```python
# If connection times out, try downloading in chunks or use Drive
# Or download locally and upload to Drive
```

### Out of memory loading data
```python
# Load fewer samples
samples = get_decathlon_file_list(data_dir, max_samples=10)

# Or use MONAI's caching with lower cache_rate
```

### NIfTI files won't load
```bash
pip install nibabel
```
