"""
Test script demonstrating pyradiomics extraction with data augmentation using pyable-dataloader.

This script shows how to:
1. Create a PyableDataset with augmentation transforms
2. Generate 5 augmented versions of a sample (with rotations and translations)
3. Extract pyradiomics features from each augmented version
4. Compare the features across augmentations

Note: This requires sample NIfTI files and a manifest. For demonstration, 
we assume a manifest.json exists with at least one subject.
"""

import tempfile
import os
from pathlib import Path
import SimpleITK as sitk
import pandas as pd

# Import from pyfe dataloader
try:
    from pyfe.dataloader import PyableDataset, Compose, RandomTranslation, RandomRotation, IntensityNormalization
    from pyfe import PYRAD  # Assuming pyfe has PYRAD class for extraction
    DATALOADER_AVAILABLE = True
except ImportError:
    print("pyfe dataloader or PYRAD not available. This is a demonstration script.")
    DATALOADER_AVAILABLE = False

def create_sample_manifest():
    """Create a sample manifest for testing. In practice, use your actual data."""
    # This is just for demonstration - replace with your actual manifest
    manifest = {
        "sample_001": {
            "images": ["/path/to/your/t1.nii.gz", "/path/to/your/t2.nii.gz"],  # Replace with real paths
            "rois": ["/path/to/your/brain_mask.nii.gz"],  # Replace with real path
            "labelmaps": [],
            "reference": 0,
            "label": 1.0
        }
    }
    return manifest

def get_augmented_files_for_pyfe(dataset, subject_idx, aug_transforms, num_augmentations=5):
    """
    Generate multiple augmented versions of a sample and return file paths.

    Args:
        dataset: PyableDataset instance
        subject_idx: Index of subject in dataset
        aug_transforms: Augmentation transforms to apply
        num_augmentations: Number of augmented versions to generate

    Returns:
        List of dicts, each containing file paths for one augmentation
    """
    augmented_files = []

    for aug_idx in range(num_augmentations):
        # Set seed for reproducibility per augmentation
        import random
        import numpy as np
        random.seed(aug_idx)
        np.random.seed(aug_idx)

        sample = dataset.get_numpy_item(subject_idx, as_nifti=True, transforms=aug_transforms)

        tmpdir = Path(tempfile.mkdtemp(prefix=f"pyfe_aug_{dataset.ids[subject_idx]}_{aug_idx}_"))
        tmpdir.mkdir(parents=True, exist_ok=True)

        out = {'augmentation_id': aug_idx, 'images': [], 'rois': [], 'labelmaps': [], 'meta': sample.get('meta', {})}

        for i, sitk_img in enumerate(sample['images']):
            p = tmpdir / f"image_{i}.nii.gz"
            sitk.WriteImage(sitk_img, str(p))
            out['images'].append(str(p))

        for i, sitk_roi in enumerate(sample.get('rois', [])):
            p = tmpdir / f"roi_{i}.nii.gz"
            sitk.WriteImage(sitk_roi, str(p))
            out['rois'].append(str(p))

        for i, sitk_lm in enumerate(sample.get('labelmaps', [])):
            p = tmpdir / f"labelmap_{i}.nii.gz"
            sitk.WriteImage(sitk_lm, str(p))
            out['labelmaps'].append(str(p))

        augmented_files.append(out)

    return augmented_files

def extract_pyradiomics_features(image_paths, roi_paths, labelmaps_paths=None):
    """
    Extract pyradiomics features from augmented files.

    Args:
        image_paths: List of paths to image files
        roi_paths: List of paths to ROI files
        labelmaps_paths: List of paths to labelmap files (optional)

    Returns:
        Dict of extracted features
    """
    if not DATALOADER_AVAILABLE:
        # Mock extraction for demonstration
        return {"mock_feature_1": 0.5, "mock_feature_2": 1.2}

    # Initialize PYRAD extractor
    extractor = PYRAD()

    # Configure features to extract (example: first-order and shape)
    extractor.setFeatureGroups({
        "FOS": {"min": 0, "max": 5000, "bin": 128},
        "SHAPE": {}
    })

    # Extract features
    features = extractor.extract(
        images=image_paths,
        roi=roi_paths[0] if roi_paths else None,  # Use first ROI
        labelmap=labelmaps_paths[0] if labelmaps_paths else None
    )

    return features

def main():
    if not DATALOADER_AVAILABLE:
        print("Required packages not available. This is a code demonstration.")
        return

    # Create sample manifest (replace with your actual data)
    manifest = create_sample_manifest()

    # Save manifest to file
    import json
    with open('sample_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    # Define augmentation transforms (rotations and translations as requested)
    augmentation_transforms = Compose([
        IntensityNormalization(method='zscore'),  # Normalize before augmentation
        RandomTranslation(
            translation_range=[[-5, 5], [-5, 5], [-2, 2]],  # ±5mm in X/Y, ±2mm in Z
            p=1.0  # Always apply
        ),
        RandomRotation(
            rotation_range_deg=[[-15, 15], [-15, 15], [-10, 10]],  # ±15° in X/Y, ±10° in Z
            p=1.0  # Always apply
        )
    ])

    # Create dataset
    dataset = PyableDataset(
        manifest='sample_manifest.json',
        target_size=[64, 64, 64],  # Example size
        target_spacing=2.0,
        # Note: transforms are applied on-demand in get_numpy_item
    )

    # Process first subject
    subject_idx = 0

    print(f"Processing subject: {dataset.ids[subject_idx]}")
    print("Generating 5 augmented versions with rotations and translations...")

    # Generate 5 augmented versions
    augmented_files = get_augmented_files_for_pyfe(
        dataset, subject_idx, augmentation_transforms, num_augmentations=5
    )

    # Extract features from each augmentation
    all_features = []

    for aug_files in augmented_files:
        print(f"Extracting features from augmentation {aug_files['augmentation_id']}...")

        features = extract_pyradiomics_features(
            image_paths=aug_files['images'],
            roi_paths=aug_files['rois'],
            labelmaps_paths=aug_files['labelmaps']
        )

        # Add augmentation ID to features
        features['augmentation_id'] = aug_files['augmentation_id']
        all_features.append(features)

        # Clean up temporary files
        tmpdir = Path(aug_files['images'][0]).parent
        import shutil
        shutil.rmtree(tmpdir)

    # Convert to DataFrame for analysis
    features_df = pd.DataFrame(all_features)

    print("\nExtracted features shape:", features_df.shape)
    print("Feature columns:", list(features_df.columns[:10]), "..." if len(features_df.columns) > 10 else "")

    # Calculate feature variability across augmentations
    numeric_features = features_df.select_dtypes(include=[float, int]).drop(columns=['augmentation_id'], errors='ignore')
    variability = numeric_features.std() / numeric_features.mean().abs()  # Coefficient of variation

    print(f"\nFeature variability across 5 augmentations:")
    print(f"Mean CV: {variability.mean():.4f}")
    print(f"Max CV: {variability.max():.4f} (feature: {variability.idxmax()})")
    print(f"Min CV: {variability.min():.4f} (feature: {variability.idxmin()})")

    # Save results
    features_df.to_csv('augmentation_features_comparison.csv', index=False)
    print("\nResults saved to 'augmentation_features_comparison.csv'")

    print("\nTest completed successfully!")
    print("This demonstrates pyradiomics extraction with 5 data augmentations (rotations + translations).")

if __name__ == "__main__":
    main()