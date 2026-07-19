# Artificial Intelligence and Computational Medicine - A Hands-on Approach

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/medAI-hands-on/blob/main/notebooks/01_medical_imaging.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository accompanies the book chapter *"Artificial Intelligence and Computational Medicine - A Hands-on Approach"* for the second edition of **Medical Applications of Artificial Intelligence** (CRC Press/Taylor & Francis, 2026).

## Overview

This chapter provides a hands-on introduction to applying AI methods in computational medicine through three interconnected domains:

1. **Medical Imaging**: Brain tumor segmentation using deep learning (U-Net)
2. **Multimodal Data Integration**: Combining imaging and clinical features for patient stratification
3. **AI-Assisted Computing**: Leveraging large language models for medical text analysis

All examples are designed to be **reproducible** and can run both **locally** and on **Google Colab**.

## Quick Start

### Option 1: Google Colab (Recommended for beginners)

Click the badge below to open the notebooks directly in Google Colab:

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| Medical Imaging | Brain tumor segmentation with U-Net | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/medAI-hands-on/blob/main/notebooks/01_medical_imaging.ipynb) |
| Multimodal Integration | Imaging + clinical data fusion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/medAI-hands-on/blob/main/notebooks/02_multimodal_integration.ipynb) |
| LLM-Assisted Computing | Medical text analysis with LLMs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/medAI-hands-on/blob/main/notebooks/03_llm_assisted_computing.ipynb) |

**Colab Tips:**
- Go to `Runtime > Change runtime type > GPU` for faster execution
- The first cell in each notebook handles installation automatically

> **Note for Private Repositories:** If this repository is private, Colab's automatic cloning requires authentication. You may need to:
> 1. Generate a GitHub Personal Access Token with `repo` scope
> 2. Use `!git clone https://<TOKEN>@github.com/arvidl/medAI-hands-on.git` in Colab
> 3. Or make the repository public for seamless Colab access

### Option 2: Local Installation

Choose **either** [uv](https://docs.astral.sh/uv/) (`pyproject.toml`) **or** conda (`environment.yml`). You do not need both.

#### Prerequisites
- Python 3.10–3.12
- NVIDIA GPU with CUDA (optional, but recommended) **or** Apple Silicon with Metal (MPS)
- For uv: install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For conda: Anaconda, Miniconda, or Miniforge

#### A) Setup with uv (recommended if you want a fast, lockable venv)

```bash
git clone https://github.com/arvidl/medAI-hands-on.git
cd medAI-hands-on

# Create .venv, install dependencies, and install this repo (editable)
uv sync

# Activate (optional; `uv run` works without activating)
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate         # Windows

# Register a Jupyter kernel (once)
python -m ipykernel install --user --name medai-handson --display-name "Python (medai-handson)"

# Launch notebooks
jupyter notebook
# or without activating:
# uv run jupyter notebook
```

Useful uv commands:

```bash
uv sync                  # create/update .venv from pyproject.toml
uv sync --extra extras   # Notebook 02: torchinfo + torch-geometric (see below)
uv add <package>         # add a dependency to pyproject.toml
uv run python scripts/create_brats_subset.py --help
uv run jupyter lab
```

**CUDA with uv:** default PyPI `torch` wheels often include CUDA on Linux. To pin a specific CUDA build:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Apple Silicon:** `uv sync` is enough; PyTorch will use MPS when available.

**If your prompt shows `(base)` (conda):** `uv sync` is still fine — it installs into project-local `.venv`, not into conda. To avoid mixing environments when you run notebooks or scripts, either:

```bash
conda deactivate          # leave conda base, then: source .venv/bin/activate
# or keep (base) and always use:
uv run jupyter notebook
uv run python -c "import torch; print(torch.__version__)"
```

Do **not** run `conda install` into `(base)` for this project if you are using uv; pick one stack (uv **or** the `medai-handson` conda env).

#### B) Setup with conda (`medai-handson`)

```bash
git clone https://github.com/arvidl/medAI-hands-on.git
cd medAI-hands-on

# Create the named environment from environment.yml
conda env create -f environment.yml

# Activate
conda activate medai-handson

# Launch notebooks
jupyter notebook
```

Useful conda commands:

```bash
conda activate medai-handson
conda env update -f environment.yml --prune   # refresh after pulling changes
conda deactivate
conda env remove -n medai-handson             # delete the env
```

**CUDA with conda** (after `conda activate medai-handson`):

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Apple Silicon:** use the default `environment.yml` as-is; conda-forge PyTorch uses MPS.

#### Verify the install

```bash
python -c "import torch, monai, nibabel, transformers; from src import data_utils, models; print(torch.__version__, torch.cuda.is_available())"
```

#### Optional: MedSAM2 (Notebook 01, Section 8)

MedSAM2 is **not** required for the main BraTS U-Net tutorial. Without it, Notebook 01 still runs and shows a **simulated** promptable-segmentation demo. For **real** MedSAM2 inference:

```bash
cd medAI-hands-on   # repo root

# Clone into the repo (gitignored — not committed)
git clone https://github.com/bowang-lab/MedSAM2.git

# Install into the same environment you use for the notebooks
# uv:
uv pip install -e ./MedSAM2
# conda (after: conda activate medai-handson):
# pip install -e ./MedSAM2

# Download weights (~300 MB) into MedSAM2/checkpoints/
cd MedSAM2 && bash download.sh && cd ..
# Or from Hugging Face: wanglab/MedSAM2 → MedSAM2_latest.pt
```

Then **restart the Jupyter kernel**, re-run Notebook 01 through the MedSAM2 availability cell, and confirm you see `✓ MedSAM2 available`.

Expect `MedSAM2/checkpoints/MedSAM2_latest.pt`. The notebook looks for `./MedSAM2` or `../MedSAM2` relative to the Jupyter working directory.

> **Note:** `MedSAM2` is installed with `uv pip install -e ./MedSAM2` (not via `pyproject.toml`). Running `uv sync` afterward may remove it from `.venv`; re-run the editable install if needed.

#### Optional: Notebook 02 extras (`torchinfo`, PyTorch Geometric)

Notebook 02’s fusion, PSN, missing-modality, calibration, and DCA sections run without these packages. To execute **all** cells — including the richer model summary and the GNN section that sets `HAS_TORCH_GEOMETRIC = True` — you need `torchinfo` and `torch-geometric`.

**uv:** these are *not* in the default `uv sync`; install the optional extras group:

```bash
cd medAI-hands-on
uv sync --extra extras
# or: uv pip install torchinfo torch-geometric
```

**conda:** `environment.yml` already includes `torchinfo` and `torch-geometric`, so a fresh `conda env create -f environment.yml` has them. If you created the env before they were added:

```bash
conda activate medai-handson
conda env update -f environment.yml --prune
# or: pip install torchinfo torch-geometric
```

Then **restart the Jupyter kernel** and re-run Notebook 02. You should see:

- a `torchinfo` model summary (instead of the basic `print(model)` fallback)
- `✓ PyTorch Geometric ... available` and `HAS_TORCH_GEOMETRIC = True`

Verify (uv):

```bash
uv run python -c "import torchinfo, torch_geometric; print(torchinfo.__version__, torch_geometric.__version__)"
```

## Repository Structure

```
medAI-hands-on/
+-- README.md                 # This file
+-- LICENSE                   # MIT License
+-- pyproject.toml            # uv / pip project + dependencies
+-- environment.yml           # Conda environment (name: medai-handson)
|
+-- assets/                   # Notebook assets (images, figures)
|   +-- unirg_RL_framework_2026.png
|
+-- chapter/                  # LaTeX source for the book chapter
|   +-- main.tex              # Main document (self-contained)
|   +-- main.pdf              # Compiled chapter
|   +-- references.bib        # Bibliography
|   +-- krantz.cls            # CRC Press document class
|   +-- figures/              # Chapter figures (PDFs) + generate_figures.py
|       +-- README.md
|       +-- generate_figures.py
|
+-- notebooks/                # Jupyter notebooks
|   +-- 01_medical_imaging.ipynb
|   +-- 02_multimodal_integration.ipynb
|   +-- 03_llm_assisted_computing.ipynb
|
+-- src/                      # Python source modules
|   +-- __init__.py
|   +-- data_utils.py         # Data loading utilities
|   +-- visualization.py      # Plotting functions
|   +-- models.py             # Neural network architectures
|
+-- models/                   # Pretrained models
|   +-- pretrained/           # Downloaded/trained model weights (~87 MB)
|       +-- brain_tumor_unet3d.pt
|       +-- README.md
|
+-- data/                     # Data directory (download-on-demand; see data/README.md)
|   +-- README.md             # Data documentation (v1.1-data subset ~900 MB)
|
+-- scripts/                  # Utility scripts
    +-- create_brats_subset.py
```

## Notebooks Overview

### 1. Medical Imaging: Brain Tumor Segmentation

**File:** `notebooks/01_medical_imaging.ipynb`

Demonstrates brain tumor segmentation using a 3D U-Net architecture with clinical context and didactic explanations:

**Core Topics:**
- Loading and visualizing multiparametric MRI data from BraTS (100 cases: 70 train / 15 validation / 15 test)
- Understanding clinical motivation for automated brain tumor segmentation
- Implementing a 3D U-Net CNN for volumetric segmentation
- Training with Dice loss to handle class imbalance in medical imaging
- Evaluating with medical imaging metrics (Dice score, sensitivity, specificity)
- Hold-out test set evaluation for clinical translation

**Learning Features:**
- Clinical interpretation guidelines for each visualization
- Review questions for self-assessment
- Suggestions for production improvements

**Runtime:** Apple Silicon M4 Max (MPS) ~ 30 min with pretrained model; NVIDIA RTX A5000 Laptop GPU ~ 2 hr 30 min including model training

<details>
<summary>Training on Apple Silicon M4 Max: >> 8 hrs</summary>

```bash
============================================================
TRAINING STARTED
============================================================
Epochs: 100 (with early stopping, patience=15)
Training samples: 70
Validation samples: 15
Test samples (held out): 15
------------------------------------------------------------
Epoch   1/100 | Train Loss: 0.9519 | Val Loss: 0.9071 | Val Dice: 0.0329 | LR: 1.00e-03
Epoch   5/100 | Train Loss: 0.5866 | Val Loss: 0.6539 | Val Dice: 0.0435 | LR: 1.00e-03
Epoch  10/100 | Train Loss: 0.5675 | Val Loss: 0.6111 | Val Dice: 0.0414 | LR: 1.00e-03
Epoch  15/100 | Train Loss: 0.5403 | Val Loss: 0.6262 | Val Dice: 0.0653 | LR: 1.00e-03
Epoch  20/100 | Train Loss: 0.5294 | Val Loss: 0.6046 | Val Dice: 0.1262 | LR: 1.00e-03
Epoch  25/100 | Train Loss: 0.3804 | Val Loss: 0.4250 | Val Dice: 0.8026 | LR: 1.00e-03
Epoch  30/100 | Train Loss: 0.3578 | Val Loss: 0.4444 | Val Dice: 0.8055 | LR: 1.00e-03
Epoch  35/100 | Train Loss: 0.3429 | Val Loss: 0.4194 | Val Dice: 0.8098 | LR: 5.00e-04
Epoch  40/100 | Train Loss: 0.3285 | Val Loss: 0.4209 | Val Dice: 0.8024 | LR: 2.50e-04
```

</details>

<details>
<summary>Training on NVIDIA RTX A5000 Laptop GPU: ~2 hrs</summary>

```bash
============================================================
TRAINING STARTED
============================================================
Epochs: 100 (with early stopping, patience=15)
Training samples: 70
Validation samples: 15
Test samples (held out): 15
------------------------------------------------------------
Epoch   1/100 | Train Loss: 0.9515 | Val Loss: 0.8988 | Val Dice: 0.0348 | LR: 1.00e-03
Epoch   5/100 | Train Loss: 0.5846 | Val Loss: 0.6584 | Val Dice: 0.0817 | LR: 1.00e-03
Epoch  10/100 | Train Loss: 0.5632 | Val Loss: 0.6114 | Val Dice: 0.0769 | LR: 1.00e-03
Epoch  15/100 | Train Loss: 0.5641 | Val Loss: 0.6162 | Val Dice: 0.0836 | LR: 1.00e-03
Epoch  20/100 | Train Loss: 0.5299 | Val Loss: 0.6024 | Val Dice: 0.1236 | LR: 1.00e-03
Epoch  25/100 | Train Loss: 0.3816 | Val Loss: 0.4260 | Val Dice: 0.8087 | LR: 1.00e-03
Epoch  30/100 | Train Loss: 0.3597 | Val Loss: 0.4596 | Val Dice: 0.8103 | LR: 1.00e-03
Epoch  35/100 | Train Loss: 0.3414 | Val Loss: 0.4137 | Val Dice: 0.8138 | LR: 5.00e-04
Epoch  40/100 | Train Loss: 0.3267 | Val Loss: 0.4125 | Val Dice: 0.8109 | LR: 2.50e-04
Epoch  45/100 | Train Loss: 0.3239 | Val Loss: 0.4137 | Val Dice: 0.8092 | LR: 2.50e-04
Epoch  50/100 | Train Loss: 0.3186 | Val Loss: 0.4115 | Val Dice: 0.8070 | LR: 2.50e-04
Epoch  55/100 | Train Loss: 0.3153 | Val Loss: 0.4123 | Val Dice: 0.8179 | LR: 2.50e-04
Epoch  60/100 | Train Loss: 0.3110 | Val Loss: 0.4088 | Val Dice: 0.8108 | LR: 2.50e-04
Epoch  65/100 | Train Loss: 0.3061 | Val Loss: 0.4038 | Val Dice: 0.8128 | LR: 2.50e-04
Epoch  70/100 | Train Loss: 0.2958 | Val Loss: 0.4106 | Val Dice: 0.8134 | LR: 1.25e-04
------------------------------------------------------------
⚡ Early stopping triggered at epoch 71
...
  File size: ~87 MB  (models/pretrained/brain_tumor_unet3d.pt)
============================================================
CPU times: user 4h 13min 11s, sys: 16min 26s, total: 4h 29min 37s
Wall time: 2h 3min 10s
```

</details>


### 2. Multimodal Data Integration

**File:** `notebooks/02_multimodal_integration.ipynb`

Combines imaging-derived features with clinical data for patient outcome prediction and stratification (Chapter §22.4):

**Core Topics:**
- Generating synthetic multimodal datasets with correlated clinical/imaging features
- **Adaptive feature fusion**: Learned modality weights α (imaging) and β (clinical)
- Training with **modality dropout** so the model can shift weight when a modality is missing
- **Missing-modality / ablation evaluation**: Accuracy and mean α/β with imaging or clinical zeroed
- **Calibration & decision curve analysis (DCA)**: Reliability curve, ECE, net benefit vs treat-all/none
- **Patient Similarity Networks (PSN)**: Cosine similarity graphs, threshold sweeps, hierarchical clustering
- **Community Detection**: Louvain modularity and permutation significance testing
- **Optional GNN section**: GCN on the PSN with semi-supervised label fractions (`uv sync --extra extras`)
- Final held-out test metrics (accuracy, precision/recall, confusion matrix)

**Learning Features:**
- "How to read this plot" guidance for each visualization
- Clinical implications and trade-offs throughout
- Collapsible technical notes (e.g., Louvain algorithm details)
- Limitations and caveats discussion

**Runtime:** ~30–45 minutes (longer if you run the optional GNN cells)

### 3. AI-Assisted Computing with LLMs

**File:** `notebooks/03_llm_assisted_computing.ipynb`

Leverages open-source Hugging Face models for medical research workflows (Chapter §22.5). No proprietary API keys required for the demos.

**Core Topics:**
- Evidence-based LLM maturity assessment (2025–2026 literature + prospective 2026 matrix)
- **Summarization**: `facebook/bart-large-cnn` loaded directly (transformers v5 removed the `summarization` pipeline)
- **Zero-shot classification**: `facebook/bart-large-mnli` via `pipeline("zero-shot-classification")`
- **Biomedical NER**: `d4data/biomedical-ner-all` via `pipeline("token-classification")`
- **Code generation**: `HuggingFaceTB/SmolLM2-360M-Instruct` drafts helpers; curated templates shown as post-review references
- **Extractive QA**: DistilBERT/SQuAD via `AutoModelForQuestionAnswering` (v5 removed `question-answering` pipeline)
- **RAG**: ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) retrieval; grounded answers from the same instruct model
- **Hallucination mitigations**: RAG grounding, chain-of-thought prompting, output-variability demo
- Best practices / governance for clinical AI (human-in-the-loop)

**Learning Features:**
- Connection to Notebooks 01 and 02 ("Three Pillars of Medical AI")
- Clinical context and expandable technical notes for each NLP task
- First run downloads several HF models (cache often ~3–4 GB)

**Runtime:** ~30–45 minutes (first run longer while models download)

## Chapter Compilation

To compile the LaTeX chapter:

```bash
cd chapter
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using latexmk:

```bash
cd chapter
latexmk -pdf main.tex
```

## Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.0+ | Deep learning framework |
| MONAI | 1.3+ | Medical imaging |
| Transformers | 4.35+ | Language models |
| NumPy | 1.24+ | Numerical computing |
| Matplotlib | 3.7+ | Visualization |
| scikit-learn | 1.3+ | Machine learning utilities |

### Additional Libraries

| Library | Purpose |
|---------|---------|
| nibabel | NIfTI file handling |
| networkx | Graph/network analysis (required for Notebook 02) |
| python-louvain | Community detection (required for Notebook 02) |
| seaborn | Statistical visualization |
| scipy | Hierarchical clustering, scientific computing |
| chromadb | Vector database for RAG (Notebook 03) |
| sentence-transformers | Embeddings for RAG retrieval (Notebook 03) |
| accelerate | Efficient Hugging Face model loading (Notebook 03) |
| torchinfo / torch-geometric | Optional Notebook 02 extras (`uv sync --extra extras`) |

## Citation

If you use this code in your research, please cite:

```bibtex
@incollection{lundervold2026ai,
  title={Artificial Intelligence and Computational Medicine: A Hands-on Approach},
  author={Lundervold, Arvid},
  booktitle={Medical Applications of Artificial Intelligence},
  edition={2nd},
  publisher={CRC Press/Taylor \& Francis},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BraTS Challenge organizers for the brain tumor segmentation benchmark
- Hugging Face for the Transformers library
- MONAI Consortium for the medical imaging framework

## Contact

**Arvid Lundervold**  
Department of Biomedicine  
University of Bergen, Norway

---

*This repository is under active development. Please report issues or suggestions via GitHub Issues.*
