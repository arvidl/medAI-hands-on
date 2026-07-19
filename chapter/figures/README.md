# Chapter Figures

PDF figures used by `chapter/main.tex` (§§22.1–22.5). Filenames must match the `\includegraphics` paths in the chapter.

## Figures

| File | Section | Description |
|------|---------|-------------|
| `fig1_workflow.pdf` | Intro | Three domains → three notebooks |
| `fig2_brain_mri.pdf` | 22.3 | Multiparametric MRI modalities (T1, T1-Gd, T2, FLAIR) |
| `fig3_segmentation.pdf` | 22.3 | Ground truth vs U-Net predictions + Dice |
| `fig4_multimodal.pdf` | 22.4 | Fusion architecture + patient similarity network |
| `fig5_llm_pipeline.pdf` | 22.5 | LLM-assisted workflow with human validation |

## Regenerating figures

Use `generate_figures.py` (writes into this directory when run via `main()`):

```bash
cd chapter/figures
# From repo root with the project env:
uv run python generate_figures.py
```

**Requirements for real (non-placeholder) fig2/fig3:**

- Decathlon or extracted subset under `data/decathlon_brain/` or compatible BraTS paths used by the script
- Pretrained weights: `models/pretrained/brain_tumor_unet3d.pt`
- Packages from `pyproject.toml` / `environment.yml` (`nibabel`, `torch`, `matplotlib`, `networkx`, `scikit-learn`)

Schematic figures (fig1, fig4 panel A, fig5) do not need imaging data. Fig4 panel B uses `src.data_utils.create_multimodal_dataset`.

To regenerate **without overwriting** the committed PDFs, change into a temporary output directory and call the figure functions after importing the module (so `savefig` lands there), or copy the results afterward.

## Specs (CRC Press / Taylor & Francis)

- **Format:** PDF preferred (vector text); PNG at 300+ DPI acceptable
- **Width:** Single column ~3.5 in, full width ~7 in
- **Colors:** Readable in grayscale
- **Labels:** Panel labels (A, B, …) for multi-panel figures
