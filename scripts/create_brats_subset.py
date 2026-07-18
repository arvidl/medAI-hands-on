#!/usr/bin/env python3
"""
Script to create BRATS subset zip file for GitHub Releases.

This script extracts the first N cases from the full Medical Segmentation
Decathlon brain tumor dataset and creates a zip file suitable for hosting
as a GitHub Release asset.

Usage:
    python create_brats_subset.py --input /path/to/Task01_BrainTumour --output brats_subset_100.zip --num-cases 100

Prerequisites:
    - Download the full Decathlon brain tumor dataset first
    - Run: python -c "from src.data_utils import download_decathlon_brain_tumor; download_decathlon_brain_tumor('data')"
"""

import argparse
import zipfile
from pathlib import Path
import shutil


def create_subset(input_dir: Path, output_zip: Path, num_cases: int = 100):
    """
    Create a subset zip from the Decathlon brain tumor dataset.
    
    Args:
        input_dir: Path to Task01_BrainTumour directory
        output_zip: Output zip file path
        num_cases: Number of cases to include (default: 100)
    """
    images_dir = input_dir / "imagesTr"
    labels_dir = input_dir / "labelsTr"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Get sorted list of cases (excluding macOS metadata files)
    image_files = sorted([
        f for f in images_dir.glob("BRATS_*.nii.gz") 
        if not f.name.startswith("._")
    ])
    
    if len(image_files) < num_cases:
        print(f"Warning: Only {len(image_files)} cases available, using all of them")
        num_cases = len(image_files)
    
    # Select first N cases
    selected_images = image_files[:num_cases]
    
    print(f"Creating subset with {num_cases} cases...")
    print(f"First case: {selected_images[0].name}")
    print(f"Last case: {selected_images[-1].name}")
    
    # Create zip file
    # Structure: Task01_BrainTumour/imagesTr/*.nii.gz
    #            Task01_BrainTumour/labelsTr/*.nii.gz
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img_path in selected_images:
            case_name = img_path.name
            label_path = labels_dir / case_name
            
            # Add image file
            arcname_img = f"Task01_BrainTumour/imagesTr/{case_name}"
            print(f"  Adding: {arcname_img}")
            zf.write(img_path, arcname_img)
            
            # Add label file if exists
            if label_path.exists():
                arcname_lbl = f"Task01_BrainTumour/labelsTr/{case_name}"
                zf.write(label_path, arcname_lbl)
            else:
                print(f"  Warning: No label found for {case_name}")
    
    # Get file size
    size_bytes = output_zip.stat().st_size
    size_gb = size_bytes / (1024 ** 3)
    
    print(f"\n✓ Created: {output_zip}")
    print(f"  Size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
    print(f"  Cases: {num_cases}")
    print(f"\nTo upload as GitHub Release:")
    print(f"  1. Go to https://github.com/arvidl/BMED365-2026/releases/new")
    print(f"  2. Tag: v1.1-data")
    print(f"  3. Title: BRATS Subset v1.1 (100 cases)")
    print(f"  4. Attach: {output_zip.name}")
    print(f"  5. Publish release")


def main():
    parser = argparse.ArgumentParser(
        description="Create BRATS subset zip for GitHub Releases"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/decathlon_brain/Task01_BrainTumour"),
        help="Path to Task01_BrainTumour directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("brats_subset_100.zip"),
        help="Output zip file path"
    )
    parser.add_argument(
        "--num-cases", "-n",
        type=int,
        default=100,
        help="Number of cases to include (default: 100)"
    )
    
    args = parser.parse_args()
    
    create_subset(args.input, args.output, args.num_cases)


if __name__ == "__main__":
    main()
