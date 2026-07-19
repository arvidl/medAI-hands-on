# Chapter Figures

This directory contains figures for the book chapter. The figures should be created as PDF or high-resolution PNG files.

## Required Figures

### Figure 1: Chapter Workflow (`fig1_workflow.pdf`)

**Description:** Overview diagram showing the three domains covered in the chapter.

**Content:**
- Three connected boxes: "Medical Imaging" → "Multimodal Integration" → "AI-Assisted Computing"
- Below each box: corresponding notebook names
- Arrows showing data flow between components
- Repository GitHub link at bottom

**Suggested tools:** draw.io, Lucidchart, or TikZ

---

### Figure 2: Brain MRI Modalities (`fig2_brain_mri.pdf`)

**Description:** Four-panel figure showing different MRI modalities.

**Content:**
- Panel A: T1-weighted axial slice
- Panel B: T1-weighted with contrast (T1ce)
- Panel C: T2-weighted
- Panel D: FLAIR
- All showing same slice with visible tumor

**Source:** Generate from notebook `01_medical_imaging.ipynb` or use BraTS dataset examples.

---

### Figure 3: Segmentation Results (`fig3_segmentation.pdf`)

**Description:** Comparison of ground truth vs. predicted segmentation.

**Content:**
- Row 1: Input MRI (FLAIR)
- Row 2: Ground truth segmentation overlay
- Row 3: Model prediction overlay
- Include Dice scores in caption

**Source:** Generate from notebook `01_medical_imaging.ipynb`

---

### Figure 4: Multimodal Integration (`fig4_multimodal.pdf`)

**Description:** Architecture diagram and patient similarity network.

**Content:**
- Left panel: Neural network architecture showing:
  - Imaging encoder
  - Clinical encoder
  - Adaptive fusion
  - Classification head
- Right panel: Patient similarity network visualization
  - Nodes colored by outcome
  - Edge thickness by similarity

**Source:** Generate from notebook `02_multimodal_integration.ipynb`

---

### Figure 5: LLM Pipeline (`fig5_llm_pipeline.pdf`)

**Description:** Workflow for LLM-assisted medical text analysis.

**Content:**
- Flow diagram: Documents → Preprocessing → LLM → Validation → Output
- Include icons for each step
- Note about human validation requirement

**Suggested tools:** draw.io, Lucidchart, or manually in LaTeX with TikZ

---

## Figure Specifications

For CRC Press/Taylor & Francis:

- **Format:** PDF preferred, PNG acceptable (300+ DPI)
- **Width:** Single column: 3.5 inches, Full width: 7 inches
- **Fonts:** Match document fonts where possible
- **Colors:** Ensure readability in grayscale
- **Labels:** Include panel labels (A, B, C, D) for multi-panel figures

## Generating Figures from Notebooks

Most figures can be generated programmatically:

```python
# In Jupyter notebook
fig.savefig('../chapter/figures/fig2_brain_mri.pdf', 
            dpi=300, bbox_inches='tight')
```

## Placeholder Files

Until final figures are created, the chapter uses placeholder boxes with descriptions in the LaTeX source.
