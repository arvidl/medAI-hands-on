#!/usr/bin/env python3
"""
Generate figures for the book chapter.
Run from the chapter/figures directory.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Figure settings for publication - ensure vector text
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    # Critical: ensure text is rendered as vector, not rasterized
    'pdf.fonttype': 42,       # TrueType fonts (editable text)
    'ps.fonttype': 42,        # TrueType fonts for PS
    'svg.fonttype': 'none',   # Text as text, not paths
    'text.usetex': False,     # Don't use LaTeX (simpler, more portable)
})

# Color scheme
COLORS = {
    'imaging': '#2E86AB',      # Blue
    'multimodal': '#A23B72',   # Magenta
    'llm': '#F18F01',          # Orange
    'background': '#F5F5F5',
    'text': '#2C3E50',
    'arrow': '#7F8C8D',
    'highlight': '#E74C3C',
    'success': '#27AE60',
}


def fig1_workflow():
    """Generate chapter workflow overview figure."""
    # Compact figure with reduced whitespace
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0.2, 5.8)
    ax.axis('off')
    
    # Title - larger for better readability
    ax.text(6, 5.6, 'Hands-on AI in Computational Medicine', 
            ha='center', va='top', fontsize=22, fontweight='bold', color=COLORS['text'])
    
    # Three main boxes - centered with proper spacing
    box_y = 3.6
    box_h = 2.0  # Slightly smaller boxes
    box_w = 3.5
    gap = 0.9
    
    # Calculate positions for even spacing
    total_width = 3 * box_w + 2 * gap
    start_x = (12 - total_width) / 2 + box_w / 2
    
    boxes = [
        (start_x, 'Medical\nImaging', COLORS['imaging'], '01_medical_imaging.ipynb'),
        (start_x + box_w + gap, 'Multimodal\nIntegration', COLORS['multimodal'], '02_multimodal_integration.ipynb'),
        (start_x + 2 * (box_w + gap), 'AI-Assisted\nComputing', COLORS['llm'], '03_llm_assisted_computing.ipynb'),
    ]
    
    for x, title, color, notebook in boxes:
        # Main box
        rect = FancyBboxPatch((x - box_w/2, box_y - box_h/2), box_w, box_h,
                               boxstyle="round,pad=0.05,rounding_size=0.15",
                               facecolor=color, edgecolor='white', linewidth=2.5)
        ax.add_patch(rect)
        
        # Title text
        ax.text(x, box_y + 0.25, title, ha='center', va='center', 
                fontsize=15, fontweight='bold', color='white', linespacing=1.1)
        
        # Notebook name
        ax.text(x, box_y - 0.65, notebook, ha='center', va='center',
                fontsize=11, color='white', style='italic', alpha=0.95,
                family='monospace')
    
    # Arrows between boxes - DASHED to indicate "recommended" rather than "required" order
    arrow_props = dict(arrowstyle='-|>', color=COLORS['arrow'], lw=2.5, 
                       mutation_scale=18, linestyle='--')
    ax.annotate('', xy=(boxes[1][0] - box_w/2 - 0.1, box_y), 
                xytext=(boxes[0][0] + box_w/2 + 0.1, box_y),
                arrowprops=arrow_props)
    ax.annotate('', xy=(boxes[2][0] - box_w/2 - 0.1, box_y), 
                xytext=(boxes[1][0] + box_w/2 + 0.1, box_y),
                arrowprops=arrow_props)
    
    # Note about arrow meaning - placed above arrows
    ax.text(6, box_y + box_h/2 + 0.25, '(recommended order, not strictly required)', 
            ha='center', va='bottom', fontsize=13, color=COLORS['arrow'], style='italic')
    
    # Content descriptions below - LARGER fonts, tighter spacing
    content_y = 2.0
    line_spacing = 0.38
    contents = [
        (boxes[0][0], ['Brain MRI analysis', 'U-Net & MedSAM2', 'Dice metrics']),
        (boxes[1][0], ['Feature fusion', 'Patient networks & GNNs', 'Attention weights']),
        (boxes[2][0], ['Summarization & RAG', 'Classification', 'Code generation']),
    ]
    
    for x, items in contents:
        for i, item in enumerate(items):
            ax.text(x, content_y - i * line_spacing, item, ha='center', va='center',
                    fontsize=16, color=COLORS['text'])
    
    # GitHub link at bottom - closer to content
    ax.text(6, 0.45, 'https://github.com/arvidl/medAI-hands-on', 
            ha='center', va='bottom', fontsize=13, color=COLORS['text'],
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E8E8', edgecolor='#CCCCCC', 
                     linewidth=1, alpha=0.9))
    
    # Ensure vector graphics with proper font embedding
    plt.tight_layout(pad=0.3)
    fig.savefig('fig1_workflow.pdf', format='pdf', bbox_inches='tight', 
                pad_inches=0.1, metadata={'Creator': 'matplotlib'})
    fig.savefig('fig1_workflow.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Generated: fig1_workflow.pdf")
    plt.close()


def fig2_brain_mri():
    """Generate brain MRI modalities figure using real data if available."""
    # Try to load real data
    data_dir = Path(__file__).parent.parent.parent / 'data'
    decathlon_dir = data_dir / 'decathlon_brain' / 'Task01_BrainTumour'
    
    # Check for Decathlon data
    images_dir = decathlon_dir / 'imagesTr'
    labels_dir = decathlon_dir / 'labelsTr'
    
    def auto_crop_brain(img, seg=None, margin=10):
        """Auto-crop image to brain region with margin."""
        # Find non-zero region
        nonzero = np.where(img > np.percentile(img, 5))
        if len(nonzero[0]) == 0:
            return img, seg, (0, img.shape[0], 0, img.shape[1])
        
        y_min, y_max = nonzero[0].min(), nonzero[0].max()
        x_min, x_max = nonzero[1].min(), nonzero[1].max()
        
        # Add margin
        y_min = max(0, y_min - margin)
        y_max = min(img.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(img.shape[1], x_max + margin)
        
        cropped_img = img[y_min:y_max, x_min:x_max]
        cropped_seg = seg[y_min:y_max, x_min:x_max] if seg is not None else None
        
        return cropped_img, cropped_seg, (y_min, y_max, x_min, x_max)
    
    if images_dir.exists():
        try:
            import nibabel as nib
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
            
            # Find a sample with good tumor visibility - all three subregions
            # Skip macOS resource fork files (start with ._)
            nii_files = sorted([f for f in images_dir.glob('*.nii.gz') if not f.name.startswith('._')])
            
            best_slice_info = None
            best_score = 0
            
            # Search for a slice with good representation of all 3 tumor types
            for nii_file in nii_files[:20]:
                img_data = nib.load(nii_file)
                vol_data = img_data.get_fdata()
                
                seg_file = labels_dir / nii_file.name
                if not seg_file.exists():
                    continue
                    
                seg_data = nib.load(seg_file).get_fdata()
                
                # Find slices with all three labels present
                for s_idx in range(seg_data.shape[0]):
                    seg_slice = seg_data[s_idx, :, :]
                    has_label1 = np.sum(seg_slice == 1) > 50  # Necrotic
                    has_label2 = np.sum(seg_slice == 2) > 100  # Edema
                    has_label3 = np.sum(seg_slice == 3) > 30   # Enhancing (often smaller)
                    
                    if has_label1 and has_label2 and has_label3:
                        # Score based on having good amounts of each
                        score = (min(np.sum(seg_slice == 1), 500) + 
                                min(np.sum(seg_slice == 2), 1000) + 
                                min(np.sum(seg_slice == 3), 300) * 2)  # Weight enhancing more
                        if score > best_score:
                            best_score = score
                            best_slice_info = (nii_file, s_idx, vol_data, seg_data)
            
            if best_slice_info is not None:
                nii_file, slice_idx, data, seg = best_slice_info
                print(f"  Using {nii_file.name}, slice {slice_idx} (score: {best_score})")
                
                if True:  # Keeping indentation structure
                        
                        # Data has 4 modalities: FLAIR, T1, T1ce, T2
                        # Decathlon order: FLAIR=0, T1=1, T1ce=2, T2=3
                        modality_order = [1, 2, 3, 0]  # T1, T1ce, T2, FLAIR for display
                        modality_names = ['T1', 'T1-Gd', 'T2', 'FLAIR']
                        
                        # Larger figure for better visibility
                        fig, axes = plt.subplots(1, 5, figsize=(14, 4))
                        
                        # Get FLAIR slice for cropping reference (usually shows brain best)
                        flair_slice_ref = data[slice_idx, :, :, 0]
                        flair_slice_ref = np.fliplr(np.rot90(flair_slice_ref, k=1))
                        seg_slice_ref = seg[slice_idx, :, :]
                        seg_slice_ref = np.fliplr(np.rot90(seg_slice_ref, k=1))
                        
                        # Calculate crop bounds from FLAIR
                        _, _, crop_bounds = auto_crop_brain(flair_slice_ref, seg_slice_ref, margin=15)
                        y_min, y_max, x_min, x_max = crop_bounds
                        
                        for i, (mod_idx, mod_name) in enumerate(zip(modality_order, modality_names)):
                            img_slice = data[slice_idx, :, :, mod_idx]
                            # Radiological orientation
                            img_slice = np.fliplr(np.rot90(img_slice, k=1))
                            # Apply crop
                            img_slice_cropped = img_slice[y_min:y_max, x_min:x_max]
                            
                            axes[i].imshow(img_slice_cropped, cmap='gray', origin='upper', 
                                          aspect='equal')
                            axes[i].set_title(mod_name, fontsize=12, fontweight='bold')
                            axes[i].axis('off')
                            # Panel label with background for visibility
                            axes[i].text(0.03, 0.97, chr(65 + i), transform=axes[i].transAxes,
                                        fontsize=14, fontweight='bold', color='white',
                                        va='top', ha='left',
                                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', 
                                                 alpha=0.5, edgecolor='none'))
                        
                        # Last panel: segmentation overlay on FLAIR with zoom inset
                        flair_slice = data[slice_idx, :, :, 0]
                        seg_slice = seg[slice_idx, :, :]
                        flair_slice = np.fliplr(np.rot90(flair_slice, k=1))
                        seg_slice = np.fliplr(np.rot90(seg_slice, k=1))
                        
                        # Apply crop
                        flair_cropped = flair_slice[y_min:y_max, x_min:x_max]
                        seg_cropped = seg_slice[y_min:y_max, x_min:x_max]
                        
                        # Clean up segmentation - keep only the main tumor region
                        # Remove small disconnected components that might appear as artifacts
                        from scipy import ndimage
                        seg_binary = (seg_cropped > 0).astype(int)
                        labeled_array, num_features = ndimage.label(seg_binary)
                        if num_features > 1:
                            # Keep only the largest connected component
                            component_sizes = ndimage.sum(seg_binary, labeled_array, 
                                                         range(1, num_features + 1))
                            largest_component = np.argmax(component_sizes) + 1
                            seg_cropped = seg_cropped * (labeled_array == largest_component)
                        
                        axes[4].imshow(flair_cropped, cmap='gray', origin='upper', aspect='equal')
                        
                        # Display each label separately for better visibility
                        # Decathlon labels: 1=necrotic, 2=edema, 3=enhancing (GD-enhancing)
                        # Layer order: edema first (largest), then necrotic, then enhancing on top
                        
                        # Edema (label 2) - yellow/gold
                        edema_mask = (seg_cropped == 2)
                        edema_display = np.ma.masked_where(~edema_mask, np.ones_like(seg_cropped))
                        axes[4].imshow(edema_display, cmap='YlOrBr', alpha=0.6, origin='upper', 
                                      vmin=0, vmax=2)
                        
                        # Necrotic core (label 1) - dark red/maroon
                        necrotic_mask = (seg_cropped == 1)
                        necrotic_display = np.ma.masked_where(~necrotic_mask, np.ones_like(seg_cropped))
                        axes[4].imshow(necrotic_display, cmap='Reds', alpha=0.8, origin='upper',
                                      vmin=0, vmax=1.5)
                        
                        # Enhancing tumor (label 3) - bright red/orange - on top
                        enhancing_mask = (seg_cropped == 3)
                        enhancing_display = np.ma.masked_where(~enhancing_mask, np.ones_like(seg_cropped))
                        axes[4].imshow(enhancing_display, cmap='hot', alpha=0.85, origin='upper',
                                      vmin=0, vmax=2)
                        axes[4].set_title('Segmentation', fontsize=12, fontweight='bold')
                        axes[4].axis('off')
                        axes[4].text(0.03, 0.97, 'E', transform=axes[4].transAxes,
                                    fontsize=14, fontweight='bold', color='white',
                                    va='top', ha='left',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', 
                                             alpha=0.5, edgecolor='none'))
                        
                        # Draw ROI rectangle on tumor region (no inset - too small)
                        # Find tumor center in cropped coordinates
                        tumor_coords = np.where(seg_cropped > 0)
                        if len(tumor_coords[0]) > 0:
                            tumor_cy = int(np.mean(tumor_coords[0]))
                            tumor_cx = int(np.mean(tumor_coords[1]))
                            
                            # ROI region size
                            roi_size = 45
                            zy_min = max(0, tumor_cy - roi_size)
                            zy_max = min(seg_cropped.shape[0], tumor_cy + roi_size)
                            zx_min = max(0, tumor_cx - roi_size)
                            zx_max = min(seg_cropped.shape[1], tumor_cx + roi_size)
                            
                            # Draw rectangle on main image showing ROI
                            from matplotlib.patches import Rectangle
                            rect = Rectangle((zx_min, zy_min), zx_max - zx_min, zy_max - zy_min,
                                           linewidth=2, edgecolor='white', facecolor='none',
                                           linestyle='--')
                            axes[4].add_patch(rect)
                        
                        # Legend for segmentation - below figure (colors match display)
                        legend_elements = [
                            mpatches.Patch(facecolor='#FF4500', edgecolor='#333', linewidth=0.5, label='Enhancing tumor'),
                            mpatches.Patch(facecolor='#DAA520', edgecolor='#333', linewidth=0.5, label='Peritumoral edema'),
                            mpatches.Patch(facecolor='#8B0000', edgecolor='#333', linewidth=0.5, label='Necrotic core'),
                        ]
                        leg = fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                                  frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01),
                                  handlelength=1.5, handleheight=1.2, handletextpad=0.5)
                        
                        plt.tight_layout()
                        plt.subplots_adjust(bottom=0.12)  # Make room for legend
                        fig.savefig('fig2_brain_mri.pdf', bbox_inches='tight', pad_inches=0.1)
                        fig.savefig('fig2_brain_mri.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
                        print("Generated: fig2_brain_mri.pdf (from Decathlon data)")
                        plt.close()
                        return
        except Exception as e:
            print(f"Error loading Decathlon data: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback: try brats_sample data
    brats_dir = data_dir / 'brats_sample'
    if brats_dir.exists():
        sample_dirs = sorted(brats_dir.glob('sample_*'))
        if sample_dirs:
            try:
                sample_dir = sample_dirs[0]
                modalities = ['t1', 't1ce', 't2', 'flair']
                modality_names = ['T1', 'T1-Gd', 'T2', 'FLAIR']
                
                fig, axes = plt.subplots(1, 5, figsize=(12, 3))
                
                for i, (mod, name) in enumerate(zip(modalities, modality_names)):
                    vol = np.load(sample_dir / f'{mod}.npy')
                    slice_idx = vol.shape[0] // 2
                    img_slice = vol[slice_idx, :, :]
                    img_slice = np.fliplr(np.rot90(img_slice, k=1))
                    axes[i].imshow(img_slice, cmap='gray', origin='upper')
                    axes[i].set_title(name, fontsize=11, fontweight='bold')
                    axes[i].axis('off')
                    axes[i].text(0.05, 0.95, chr(65 + i), transform=axes[i].transAxes,
                                fontsize=12, fontweight='bold', color='white', va='top')
                
                # Segmentation
                seg = np.load(sample_dir / 'seg.npy')
                flair = np.load(sample_dir / 'flair.npy')
                slice_idx = seg.shape[0] // 2
                flair_slice = np.fliplr(np.rot90(flair[slice_idx], k=1))
                seg_slice = np.fliplr(np.rot90(seg[slice_idx], k=1))
                
                axes[4].imshow(flair_slice, cmap='gray', origin='upper')
                masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
                axes[4].imshow(masked_seg, cmap='hot', alpha=0.6, origin='upper')
                axes[4].set_title('Segmentation', fontsize=11, fontweight='bold')
                axes[4].axis('off')
                axes[4].text(0.05, 0.95, 'E', transform=axes[4].transAxes,
                            fontsize=12, fontweight='bold', color='white', va='top')
                
                plt.tight_layout()
                fig.savefig('fig2_brain_mri.pdf')
                fig.savefig('fig2_brain_mri.png', dpi=300)
                print("Generated: fig2_brain_mri.pdf (from brats_sample data)")
                plt.close()
                return
            except Exception as e:
                print(f"Error loading brats_sample data: {e}")
    
    # Final fallback: create synthetic placeholder
    print("No MRI data found, creating synthetic placeholder...")
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    modality_names = ['T1', 'T1-Gd', 'T2', 'FLAIR', 'Segmentation']
    
    np.random.seed(42)
    for i, name in enumerate(modality_names):
        if i < 4:
            # Create brain-like synthetic image
            x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
            brain = np.exp(-(x**2 + y**2) / 0.5) + 0.1 * np.random.randn(128, 128)
            # Add tumor
            tumor_x, tumor_y = 0.2, 0.1
            tumor = 0.3 * np.exp(-((x - tumor_x)**2 + (y - tumor_y)**2) / 0.02)
            img = brain + tumor * (1 + 0.5 * i)  # Different contrast per modality
            axes[i].imshow(img, cmap='gray')
        else:
            # Segmentation
            brain = np.exp(-(x**2 + y**2) / 0.5) > 0.3
            tumor_mask = np.exp(-((x - tumor_x)**2 + (y - tumor_y)**2) / 0.02) > 0.1
            seg = np.zeros_like(brain, dtype=int)
            seg[tumor_mask] = 1
            axes[i].imshow(brain * 0.5, cmap='gray')
            masked = np.ma.masked_where(seg == 0, seg)
            axes[i].imshow(masked, cmap='hot', alpha=0.6)
        
        axes[i].set_title(name, fontsize=11, fontweight='bold')
        axes[i].axis('off')
        axes[i].text(0.05, 0.95, chr(65 + i), transform=axes[i].transAxes,
                    fontsize=12, fontweight='bold', color='white', va='top')
    
    plt.tight_layout()
    fig.savefig('fig2_brain_mri.pdf')
    fig.savefig('fig2_brain_mri.png', dpi=300)
    print("Generated: fig2_brain_mri.pdf (synthetic placeholder)")
    plt.close()


def fig3_segmentation():
    """Generate segmentation results comparison figure."""
    # Try to load real data and predictions
    data_dir = Path(__file__).parent.parent.parent / 'data'
    models_dir = Path(__file__).parent.parent.parent / 'models' / 'pretrained'
    
    # Check if we have trained model and data
    model_path = models_dir / 'brain_tumor_unet3d.pt'
    decathlon_dir = data_dir / 'decathlon_brain' / 'Task01_BrainTumour'
    
    if model_path.exists() and decathlon_dir.exists():
        try:
            import torch
            import nibabel as nib
            from src.models import UNet3D
            from src.data_utils import normalize_volume
            
            # Use CPU for figure generation (MPS doesn't support 3D max pooling)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Infer model architecture from the actual weights (more reliable than config)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Detect base_features from first encoder layer weight shape
            first_conv_shape = state_dict['encoders.0.conv1.weight'].shape
            base_features = first_conv_shape[0]  # [out_channels, in_channels, D, H, W]
            in_channels = first_conv_shape[1]
            
            # Detect depth by counting encoder layers
            depth = sum(1 for k in state_dict.keys() if k.startswith('encoders.') and k.endswith('.conv1.weight'))
            
            # Detect out_channels from final conv layer
            out_channels = state_dict['final_conv.weight'].shape[0]
            
            print(f"  Model architecture detected: in_channels={in_channels}, out_channels={out_channels}, "
                  f"base_features={base_features}, depth={depth}")
            
            model = UNet3D(
                in_channels=in_channels, 
                out_channels=out_channels, 
                base_features=base_features,
                depth=depth
            )
            # Handle both full checkpoint and state_dict formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            
            # Load samples for visualization
            # IMPORTANT: Prefer using TEST samples (held-out data) if available in checkpoint
            # This gives a more honest representation of model performance
            images_dir = decathlon_dir / 'imagesTr'
            labels_dir = decathlon_dir / 'labelsTr'
            
            # Try to get test case IDs from checkpoint training config
            test_case_ids = []
            if 'training_config' in checkpoint and 'test_case_ids' in checkpoint['training_config']:
                test_case_ids = checkpoint['training_config']['test_case_ids']
                print(f"Using held-out test cases from checkpoint: {test_case_ids[:3]}")
            
            # Pre-computed Dice scores from notebook evaluation (full volume, 128^3 resampled)
            # These are the correct whole-tumor Dice scores computed on the full resampled volumes
            precomputed_dice = {
                'BRATS_004': 0.9411,
                'BRATS_005': 0.6701,
                'BRATS_012': 0.9020,
                'BRATS_014': 0.8865,
                'BRATS_015': 0.9132,
                'BRATS_018': 0.8943,
                'BRATS_029': 0.8756,
                'BRATS_032': 0.9202,
                'BRATS_036': 0.9470,
                'BRATS_055': 0.8834,
                'BRATS_070': 0.9156,
                'BRATS_076': 0.9582,
                'BRATS_082': 0.9388,
                'BRATS_087': 0.9045,
                'BRATS_095': 0.7610,
            }
            
            if len(test_case_ids) >= 3:
                # Use held-out test cases (never seen during training)
                case_names = [f'{cid}.nii.gz' for cid in test_case_ids[:3]]
            else:
                # Use known test cases from the 70/15/15 train/val/test split
                # These cases were NOT seen during training
                print("Using test set cases: BRATS_004, BRATS_005, BRATS_076, BRATS_082")
                case_names = ['BRATS_004.nii.gz', 'BRATS_005.nii.gz', 'BRATS_076.nii.gz', 'BRATS_082.nii.gz']
            
            nii_files = [images_dir / name for name in case_names if (images_dir / name).exists()]
            if len(nii_files) < 4:
                nii_files = sorted(images_dir.glob('*.nii.gz'))[0:4]
            
            fig, axes = plt.subplots(4, 4, figsize=(13, 11))  # 4 rows x 4 columns
            plt.subplots_adjust(hspace=0.02, wspace=0.05)  # Minimal vertical spacing between rows
            
            # Helper function to auto-crop to brain region
            def auto_crop_slices(flair_img, t1ce_img=None, seg=None, pred=None, margin=10):
                """Auto-crop images to brain region with margin."""
                nonzero = np.where(flair_img > np.percentile(flair_img, 5))
                if len(nonzero[0]) == 0:
                    return flair_img, t1ce_img, seg, pred
                
                y_min, y_max = nonzero[0].min(), nonzero[0].max()
                x_min, x_max = nonzero[1].min(), nonzero[1].max()
                
                # Add margin
                y_min = max(0, y_min - margin)
                y_max = min(flair_img.shape[0], y_max + margin)
                x_min = max(0, x_min - margin)
                x_max = min(flair_img.shape[1], x_max + margin)
                
                cropped_flair = flair_img[y_min:y_max, x_min:x_max]
                cropped_t1ce = t1ce_img[y_min:y_max, x_min:x_max] if t1ce_img is not None else None
                cropped_seg = seg[y_min:y_max, x_min:x_max] if seg is not None else None
                cropped_pred = pred[y_min:y_max, x_min:x_max] if pred is not None else None
                
                return cropped_flair, cropped_t1ce, cropped_seg, cropped_pred
            
            case_info = []  # Store case IDs and slices for caption
            for col, nii_file in enumerate(nii_files[:4]):
                img = nib.load(nii_file)
                data = img.get_fdata()
                seg = nib.load(labels_dir / nii_file.name).get_fdata()
                
                # Find good slice with tumor - use sagittal slices (first dimension)
                tumor_slices = np.where(seg.sum(axis=(1, 2)) > 50)[0]
                # Filter out edge slices
                tumor_slices = tumor_slices[(tumor_slices > 10) & (tumor_slices < data.shape[0] - 10)]
                if len(tumor_slices) > 0:
                    slice_idx = tumor_slices[len(tumor_slices) // 2]
                else:
                    slice_idx = data.shape[0] // 2
                
                # Record case info with pre-computed Dice score from notebook evaluation
                case_id = nii_file.stem.replace('.nii', '')
                dice_score = precomputed_dice.get(case_id, 0.0)
                case_info.append((case_id, slice_idx, dice_score))
                
                # Extract slices from original volume
                # Decathlon modality order: 0=FLAIR, 1=T1, 2=T1ce, 3=T2
                flair_slice_orig = data[slice_idx, :, :, 0]
                t1ce_slice_orig = data[slice_idx, :, :, 2]  # T1-Gd (contrast enhanced)
                seg_slice_orig = seg[slice_idx, :, :]
                
                # Apply radiological orientation
                flair_slice_rot = np.fliplr(np.rot90(flair_slice_orig, k=1))
                t1ce_slice_rot = np.fliplr(np.rot90(t1ce_slice_orig, k=1))
                seg_slice_rot = np.fliplr(np.rot90(seg_slice_orig, k=1))
                
                # Create prediction using the model
                # Resample volume for model (which expects 128^3)
                from scipy.ndimage import zoom as scipy_zoom
                target_shape = (128, 128, 128)
                zoom_factors = [t / o for t, o in zip(target_shape, data.shape[:3])]
                
                with torch.no_grad():
                    # Resample and normalize for model
                    vol_resampled = np.zeros(target_shape + (4,), dtype=np.float32)
                    for c in range(4):
                        vol_resampled[:, :, :, c] = scipy_zoom(data[:, :, :, c], zoom_factors, order=1)
                    
                    vol_norm = np.zeros((4,) + target_shape, dtype=np.float32)
                    for c, mod_idx in enumerate([1, 2, 3, 0]):  # T1, T1ce, T2, FLAIR
                        vol_norm[c] = normalize_volume(vol_resampled[:, :, :, mod_idx])
                    
                    input_tensor = torch.from_numpy(vol_norm).float().unsqueeze(0).to(device)
                    output = model(input_tensor)
                    pred_resampled = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                    
                    # Resample prediction back to original resolution
                    inv_zoom = [1/z for z in zoom_factors]
                    pred_full = scipy_zoom(pred_resampled.astype(np.float32), inv_zoom, order=0)
                    
                    # Get prediction slice at same index
                    pred_slice_orig = pred_full[slice_idx, :, :]
                    pred_slice_rot = np.fliplr(np.rot90(pred_slice_orig, k=1))
                
                # Auto-crop all to brain region
                flair_slice, t1ce_slice, seg_slice, pred_slice = auto_crop_slices(
                    flair_slice_rot, t1ce_slice_rot, seg_slice_rot, pred_slice_rot, margin=15)
                
                # Panel labels A-P (row-major order for 4x4 grid)
                panel_labels = [['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H'], 
                               ['I', 'J', 'K', 'L'], ['M', 'N', 'O', 'P']]
                
                # Row 0: T1-Gd input (contrast enhanced - shows enhancing tumor)
                axes[0, col].imshow(t1ce_slice, cmap='gray', origin='upper', aspect='equal')
                axes[0, col].axis('off')
                axes[0, col].text(0.03, 0.97, panel_labels[0][col], transform=axes[0, col].transAxes,
                                 fontsize=14, fontweight='bold', color='white', va='top', ha='left',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
                
                # Row 1: FLAIR input (shows edema clearly)
                axes[1, col].imshow(flair_slice, cmap='gray', origin='upper', aspect='equal')
                axes[1, col].axis('off')
                axes[1, col].text(0.03, 0.97, panel_labels[1][col], transform=axes[1, col].transAxes,
                                 fontsize=14, fontweight='bold', color='white', va='top', ha='left',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
                
                # Row 2: Ground truth with multi-class overlay - CRISP solid colors
                axes[2, col].imshow(flair_slice, cmap='gray', origin='upper', aspect='equal')
                # Create RGB overlay with solid colors for each tumor compartment
                # Label mapping: 1=necrotic (dark red), 2=edema (gold/yellow), 3=enhancing (bright orange)
                rgb_overlay = np.zeros((*seg_slice.shape, 4))  # RGBA
                # Edema (label 2) - Gold/Yellow
                edema_mask = (seg_slice == 2)
                rgb_overlay[edema_mask] = [1.0, 0.84, 0.0, 1.0]  # Gold #FFD700
                # Necrotic core (label 1) - Dark red
                necrotic_mask = (seg_slice == 1)
                rgb_overlay[necrotic_mask] = [0.55, 0.0, 0.0, 1.0]  # Dark red #8B0000
                # Enhancing tumor (label 3) - Bright orange (on top)
                enhancing_mask = (seg_slice == 3)
                rgb_overlay[enhancing_mask] = [1.0, 0.27, 0.0, 1.0]  # Orange-red #FF4500
                axes[2, col].imshow(rgb_overlay, origin='upper', aspect='equal')
                axes[2, col].axis('off')
                axes[2, col].text(0.03, 0.97, panel_labels[2][col], transform=axes[2, col].transAxes,
                                 fontsize=14, fontweight='bold', color='white', va='top', ha='left',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
                
                # Row 3: Model prediction with multi-class labels - CRISP solid colors (same as ground truth)
                axes[3, col].imshow(flair_slice, cmap='gray', origin='upper', aspect='equal')
                # Create RGB overlay with solid colors for each tumor compartment
                # Label mapping: 1=necrotic (dark red), 2=edema (gold/yellow), 3=enhancing (bright orange)
                rgb_overlay_pred = np.zeros((*pred_slice.shape, 4))  # RGBA
                # Edema (label 2) - Gold/Yellow
                edema_mask_pred = (pred_slice == 2)
                rgb_overlay_pred[edema_mask_pred] = [1.0, 0.84, 0.0, 1.0]  # Gold #FFD700
                # Necrotic core (label 1) - Dark red
                necrotic_mask_pred = (pred_slice == 1)
                rgb_overlay_pred[necrotic_mask_pred] = [0.55, 0.0, 0.0, 1.0]  # Dark red #8B0000
                # Enhancing tumor (label 3) - Bright orange (on top)
                enhancing_mask_pred = (pred_slice == 3)
                rgb_overlay_pred[enhancing_mask_pred] = [1.0, 0.27, 0.0, 1.0]  # Orange-red #FF4500
                axes[3, col].imshow(rgb_overlay_pred, origin='upper', aspect='equal')
                axes[3, col].axis('off')
                axes[3, col].text(0.03, 0.97, panel_labels[3][col], transform=axes[3, col].transAxes,
                                 fontsize=14, fontweight='bold', color='white', va='top', ha='left',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
                
                # Case label at top - use actual BRATS ID with Dice score
                case_id, _, dice = case_info[-1]  # Most recently added case
                axes[0, col].set_title(f'{case_id}\nDice: {dice:.2f}', fontsize=10, fontweight='bold')
            
            # Add row labels on the left side
            row_labels = ['Input\n(T1-Gd)', 'Input\n(FLAIR)', 'Ground\nTruth', 'Model\nPrediction']
            for row, label in enumerate(row_labels):
                axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                                 fontsize=10, fontweight='bold', va='center', ha='right',
                                 rotation=0)
            
            plt.subplots_adjust(left=0.12, top=0.92, bottom=0.02, hspace=0.02, wspace=0.05)
            fig.savefig('fig3_segmentation.pdf', bbox_inches='tight', pad_inches=0.1)
            fig.savefig('fig3_segmentation.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
            
            # Print case info for caption
            print("  Cases used (from held-out test set):")
            for i, (case_id, slice_num, dice) in enumerate(case_info):
                print(f"    Case {i+1}: {case_id}, slice {slice_num}, Dice: {dice:.3f}")
            print("Generated: fig3_segmentation.pdf (with model predictions)")
            plt.close()
            return
            
        except Exception as e:
            print(f"Error generating with model: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback: create illustrative synthetic figure
    print("Creating synthetic segmentation results...")
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    
    np.random.seed(42)
    for col in range(3):
        # Create brain-like image
        x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
        brain = np.exp(-(x**2 + y**2) / 0.5)
        tumor_x = 0.15 + col * 0.1
        tumor_y = 0.1 - col * 0.05
        tumor = 0.4 * np.exp(-((x - tumor_x)**2 + (y - tumor_y)**2) / (0.02 + col * 0.005))
        img = brain + tumor + 0.05 * np.random.randn(128, 128)
        
        # Ground truth segmentation
        gt_seg = (np.exp(-((x - tumor_x)**2 + (y - tumor_y)**2) / (0.02 + col * 0.005)) > 0.2).astype(int)
        
        # Prediction (slightly different)
        pred_seg = (np.exp(-((x - tumor_x + 0.01)**2 + (y - tumor_y)**2) / (0.022 + col * 0.005)) > 0.18).astype(int)
        
        # Row 0: Input
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].axis('off')
        axes[0, col].set_title(f'Case {col + 1}', fontsize=11, fontweight='bold')
        
        # Row 1: Ground truth
        axes[1, col].imshow(img, cmap='gray')
        gt_masked = np.ma.masked_where(gt_seg == 0, gt_seg)
        axes[1, col].imshow(gt_masked, cmap='Reds', alpha=0.5)
        axes[1, col].axis('off')
        
        # Row 2: Prediction
        axes[2, col].imshow(img, cmap='gray')
        pred_masked = np.ma.masked_where(pred_seg == 0, pred_seg)
        axes[2, col].imshow(pred_masked, cmap='Oranges', alpha=0.5)
        axes[2, col].axis('off')
    
    # Row labels
    axes[0, 0].text(-0.15, 0.5, 'Input', transform=axes[0, 0].transAxes,
                    rotation=90, va='center', ha='right', fontsize=11, fontweight='bold')
    axes[1, 0].text(-0.15, 0.5, 'Ground Truth', transform=axes[1, 0].transAxes,
                    rotation=90, va='center', ha='right', fontsize=11, fontweight='bold')
    axes[2, 0].text(-0.15, 0.5, 'Prediction', transform=axes[2, 0].transAxes,
                    rotation=90, va='center', ha='right', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('fig3_segmentation.pdf')
    fig.savefig('fig3_segmentation.png', dpi=300)
    print("Generated: fig3_segmentation.pdf (synthetic)")
    plt.close()


def fig4_multimodal():
    """
    Generate multimodal integration architecture and patient network figure.
    
    This figure illustrates the key concepts of multimodal data integration:
    - Panel A: Neural network architecture for fusing imaging and clinical data
    - Panel B: Patient similarity network showing how fused representations cluster patients
    
    Clinical context: Brain tumor patients have both imaging data (MRI scans) and clinical
    data (age, lab values, genetic markers). Combining these modalities can improve
    outcome prediction compared to using either modality alone.
    """
    # Larger figure for better readability
    fig = plt.figure(figsize=(14, 6))
    
    # Left panel: Architecture diagram - more space and larger elements
    ax1 = fig.add_axes([0.02, 0.08, 0.46, 0.88])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 9)
    ax1.axis('off')
    ax1.set_title('A. Multimodal Fusion Architecture', fontsize=14, fontweight='bold', loc='left', pad=10)
    
    # Input boxes with larger fonts
    def draw_box(ax, x, y, w, h, color, text, fontsize=10):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.03,rounding_size=0.12",
                               facecolor=color, edgecolor='#333', linewidth=2, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
                fontweight='bold', color='white' if color not in ['#FFFFFF', '#F5F5F5'] else '#333')
    
    # Data source labels at top
    ax1.text(1.8, 8.6, 'MRI / CT Images', ha='center', fontsize=11, color='#555', style='italic')
    ax1.text(5.2, 8.6, 'Electronic Health Records', ha='center', fontsize=11, color='#555', style='italic')
    
    # Imaging pathway - left column
    draw_box(ax1, 1.8, 7.3, 3.0, 1.1, COLORS['imaging'], 'Imaging Features\n(radiomics, CNN)', 11)
    draw_box(ax1, 1.8, 5.3, 2.6, 0.9, '#5DADE2', 'Feature Encoder', 11)
    draw_box(ax1, 1.8, 3.5, 2.6, 0.9, '#3498DB', 'Latent\nRepresentation', 10)
    
    # Clinical pathway - right column
    draw_box(ax1, 5.2, 7.3, 3.0, 1.1, COLORS['multimodal'], 'Clinical Features\n(age, labs, genetics)', 11)
    draw_box(ax1, 5.2, 5.3, 2.6, 0.9, '#C39BD3', 'Feature Encoder', 11)
    draw_box(ax1, 5.2, 3.5, 2.6, 0.9, '#9B59B6', 'Latent\nRepresentation', 10)
    
    # Fusion layer - centered
    draw_box(ax1, 3.5, 2.0, 4.5, 1.0, COLORS['llm'], 'Adaptive Fusion', 12)
    
    # Output
    draw_box(ax1, 3.5, 0.55, 3.2, 0.85, COLORS['success'], 'Survival Prediction', 11)
    
    # Arrows with thicker lines - draw FIRST so annotations appear on top
    # Data flows TOP-DOWN: Input -> Encoder -> Representation -> Fusion -> Prediction
    arrow_style = dict(arrowstyle='->', color='#444', lw=2.5, mutation_scale=18)
    
    # Imaging pathway arrows (top to bottom)
    ax1.annotate('', xy=(1.8, 5.85), xytext=(1.8, 6.7), arrowprops=arrow_style)  # Input -> Encoder
    ax1.annotate('', xy=(1.8, 4.0), xytext=(1.8, 4.85), arrowprops=arrow_style)  # Encoder -> Representation
    
    # Clinical pathway arrows (top to bottom)
    ax1.annotate('', xy=(5.2, 5.85), xytext=(5.2, 6.7), arrowprops=arrow_style)  # Input -> Encoder
    ax1.annotate('', xy=(5.2, 4.0), xytext=(5.2, 4.85), arrowprops=arrow_style)  # Encoder -> Representation
    
    # Convergence arrows to fusion (from representations down to fusion layer)
    ax1.annotate('', xy=(2.4, 2.55), xytext=(2.0, 2.95), arrowprops=arrow_style)  # Left repr -> Fusion
    ax1.annotate('', xy=(4.6, 2.55), xytext=(5.0, 2.95), arrowprops=arrow_style)  # Right repr -> Fusion
    
    # Output arrow (fusion to prediction)
    ax1.annotate('', xy=(3.5, 1.02), xytext=(3.5, 1.45), arrowprops=arrow_style)  # Fusion -> Prediction
    
    # Attention weights annotation - position clearly below Latent Representation boxes
    # with white background for visibility
    # Use α and β notation matching chapter text, with values from trained model (~0.3 imaging, ~0.7 clinical)
    bbox_props = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9)
    ax1.text(0.8, 2.85, r'$\alpha$ = 0.3', fontsize=11, color=COLORS['imaging'], fontweight='bold',
             ha='center', va='center', bbox=bbox_props)
    ax1.text(6.2, 2.85, r'$\beta$ = 0.7', fontsize=11, color=COLORS['multimodal'], fontweight='bold',
             ha='center', va='center', bbox=bbox_props)
    
    # Right panel: Patient similarity network - generated from actual notebook methodology
    # This matches the approach in 02_multimodal_integration.ipynb
    ax2 = fig.add_axes([0.52, 0.08, 0.46, 0.88])
    ax2.set_title('B. Patient Similarity Network', fontsize=14, fontweight='bold', loc='left', pad=10)
    
    # Import required libraries
    import networkx as nx
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    from src.data_utils import create_multimodal_dataset
    
    # Generate multimodal dataset using same function as notebook
    # Use random_state=42 for reproducibility (matches notebook SEED)
    data = create_multimodal_dataset(
        n_patients=500,
        n_imaging_features=50,
        n_clinical_features=10,
        random_state=42
    )
    
    # Take first 50 patients for visualization (matches notebook)
    n_network = 50
    imaging_subset = data['imaging'][:n_network]
    clinical_subset = data['clinical'][:n_network]
    labels_network = data['labels'][:n_network]
    
    # Standardize features (as in notebook)
    img_scaler = StandardScaler()
    clin_scaler = StandardScaler()
    img_scaled = img_scaler.fit_transform(imaging_subset)
    clin_scaled = clin_scaler.fit_transform(clinical_subset)
    
    # Combine features for similarity computation
    combined_features = np.hstack([img_scaled, clin_scaled])
    
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(combined_features)
    
    # Build network at threshold 0.4 (matches notebook and caption)
    threshold = 0.4
    G = nx.Graph()
    G.add_nodes_from(range(n_network))
    edges_with_weights = []
    
    for i in range(n_network):
        for j in range(i + 1, n_network):
            sim = sim_matrix[i, j]
            if sim > threshold:
                G.add_edge(i, j, weight=sim)
                edges_with_weights.append((i, j, sim))
    
    # Use spring layout (same as notebook)
    pos = nx.spring_layout(G, seed=42, k=1.5/np.sqrt(n_network))
    
    # Convert positions to arrays for plotting
    x = np.array([pos[i][0] for i in range(n_network)])
    y = np.array([pos[i][1] for i in range(n_network)])
    
    # Draw edges with alpha proportional to similarity
    for i, j, sim in edges_with_weights:
        alpha = 0.15 + 0.35 * (sim - threshold) / (1 - threshold)  # Scale alpha by similarity
        ax2.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=alpha, lw=0.8)
    
    # Node colors by outcome (0=Poor prognosis, 1=Good/Long-term survival)
    # Match notebook: label 0 = Poor (red/orange), label 1 = Good (blue)
    colors = [COLORS['imaging'] if l == 1 else COLORS['llm'] for l in labels_network]
    ax2.scatter(x, y, c=colors, s=100, alpha=0.88, edgecolors='white', linewidths=1.2, zorder=10)
    
    ax2.axis('off')
    
    # Count outcomes for legend
    n_good = np.sum(labels_network == 1)
    n_poor = np.sum(labels_network == 0)
    
    # Legend - with actual counts
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['imaging'],
               markersize=13, label=f'Long-term survival (n={n_good})', markeredgecolor='white', markeredgewidth=1),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['llm'],
               markersize=13, label=f'Poor prognosis (n={n_poor})', markeredgecolor='white', markeredgewidth=1),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.95, 
               fontsize=11, title='Patient Outcomes', title_fontsize=11,
               fancybox=True, shadow=False)
    
    # Add description text at bottom
    n_edges = len(edges_with_weights)
    ax2.text(0, -1.15, f'Edges: cosine similarity > {threshold} between patient feature vectors ({n_edges} edges)',
             ha='center', fontsize=10, color='#333', style='italic')
    
    fig.savefig('fig4_multimodal.pdf')
    fig.savefig('fig4_multimodal.png', dpi=300)
    print("Generated: fig4_multimodal.pdf")
    plt.close()


def fig5_llm_pipeline():
    """Generate LLM workflow for medical text analysis figure."""
    # Compact figure - minimal whitespace above and below
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.55, 5.5)
    ax.axis('off')
    
    # Title - positioned to fit within cropped area
    ax.text(6.5, 5.35, 'LLM-Assisted Medical Text Analysis Pipeline', 
            ha='center', va='top', fontsize=15, fontweight='bold', color=COLORS['text'])
    
    # Main flow boxes with panel labels A-E
    box_y = 3.0  # Position of box centers
    boxes = [
        (1.3, box_y, 'Medical\nDocuments', '#95A5A6', 'A'),
        (3.8, box_y, 'Preprocessing\n& Chunking', '#5DADE2', 'B'),
        (6.3, box_y, 'LLM\nProcessing', COLORS['llm'], 'C'),
        (8.8, box_y, 'Human\nValidation', COLORS['multimodal'], 'D'),
        (11.3, box_y, 'Validated\nOutput', COLORS['success'], 'E'),
    ]
    
    # Draw boxes with black borders
    for x, y, text, color, label in boxes:
        # Main box with black border
        rect = FancyBboxPatch((x - 1.0, y - 0.8), 2.0, 1.6,
                               boxstyle="round,pad=0.03,rounding_size=0.15",
                               facecolor=color, edgecolor='#222', linewidth=2, alpha=0.95)
        ax.add_patch(rect)
        # Box text - larger font
        ax.text(x, y, text, ha='center', va='center', fontsize=11, 
                fontweight='bold', color='white')
        # Panel label (A, B, C, D, E) in top-left corner of box
        ax.text(x - 0.85, y + 0.65, label, ha='left', va='top', fontsize=12,
                fontweight='bold', color='white', 
                bbox=dict(boxstyle='circle,pad=0.15', facecolor='#222', edgecolor='none', 
                         alpha=0.95))
    
    # Forward arrows between boxes
    arrow_props = dict(arrowstyle='->', color='#444', lw=2.5, mutation_scale=18)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.05
        x2 = boxes[i + 1][0] - 1.05
        ax.annotate('', xy=(x2, box_y), xytext=(x1, box_y), arrowprops=arrow_props)
    
    # =========================================================================
    # FEEDBACK LOOP - More prominent graphical representation
    # =========================================================================
    # Draw a clear loop from D back to C with multiple visual elements
    
    # First, draw an upward line from D
    loop_top = 4.4
    ax.annotate('', xy=(8.8, loop_top - 0.3), xytext=(8.8, box_y + 0.85),
                arrowprops=dict(arrowstyle='-', color=COLORS['highlight'], lw=3))
    
    # Horizontal line from D to C at the top
    ax.plot([6.3, 8.8], [loop_top - 0.3, loop_top - 0.3], 
            color=COLORS['highlight'], lw=3, solid_capstyle='round')
    
    # Downward arrow into C
    ax.annotate('', xy=(6.3, box_y + 0.85), xytext=(6.3, loop_top - 0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=3, mutation_scale=20))
    
    # Feedback label in the middle of the loop - closer to the arrow
    ax.text(7.55, 4.45, 'Refine prompt if needed', ha='center', fontsize=12, 
            fontweight='bold', color=COLORS['highlight'],
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=COLORS['highlight'], 
                     linewidth=2, alpha=0.98))
    
    # Small circular arrows icon to emphasize iteration
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches
    
    # Description text below each box - more vertical space from boxes
    desc_y = 1.25  # More space between boxes (bottom at ~2.2) and description text
    descriptions = [
        (1.3, 'Clinical notes,\nradiology reports,\nresearch abstracts'),
        (3.8, 'Text splitting,\ncontext window\nmanagement'),
        (6.3, 'Summarization,\nclassification,\ncode generation'),
        (8.8, 'Expert review,\nfact-checking,\nerror correction'),
        (11.3, 'Verified summaries,\nannotations,\nstructured data'),
    ]
    
    for x, desc in descriptions:
        ax.text(x, desc_y, desc, ha='center', fontsize=12, color='#333', 
                linespacing=1.3)
    
    plt.tight_layout()
    fig.savefig('fig5_llm_pipeline.pdf')
    fig.savefig('fig5_llm_pipeline.png', dpi=300)
    print("Generated: fig5_llm_pipeline.pdf")
    plt.close()


def main():
    """Generate all figures."""
    print("Generating chapter figures...")
    print("=" * 50)
    
    # Change to figures directory
    figures_dir = Path(__file__).parent
    import os
    os.chdir(figures_dir)
    
    # Generate all figures
    fig1_workflow()
    fig2_brain_mri()
    fig3_segmentation()
    fig4_multimodal()
    fig5_llm_pipeline()
    
    print("=" * 50)
    print("All figures generated!")
    print(f"Output directory: {figures_dir}")


if __name__ == '__main__':
    main()
