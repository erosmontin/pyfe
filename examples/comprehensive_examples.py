"""
Comprehensive Examples: PyTorch Training + PyFE Feature Extraction + Data Augmentation

This guide demonstrates how to use pyfe with pyable-dataloader for:
1. PyTorch training with medical image data
2. Feature extraction with pyfe
3. Data augmentation in all scenarios
4. Customizing augmentations for different use cases

Perfect for both humans and AI agents to understand the full workflow.
"""

import os
import json
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import SimpleITK as sitk
from pathlib import Path

# Import pyfe and pyable-dataloader
from pyfe import exrtactMyFeatures
from pyfe_adapter import convert_manifest_to_pyfe

try:
    from pyable_dataloader import PyableDataset
    from pyable_dataloader.transforms import (
        Compose, IntensityNormalization, RandomTranslation, RandomFlip,
        RandomRotation, RandomScale, RandomNoise, RandomBrightness,
        RandomContrast, RandomGamma, ElasticDeformation, RandomCrop,
        CenterCrop, RandomElasticDeformation, HistogramEqualization,
        CLAHE, GaussianBlur, MedianBlur, RandomAffine
    )
    DATALOADER_AVAILABLE = True
except ImportError:
    print("Warning: pyable_dataloader not available. Some examples will be skipped.")
    DATALOADER_AVAILABLE = False


# =============================================================================
# 1. PYTORCH TRAINING WITH MEDICAL IMAGES
# =============================================================================

class MedicalImageClassifier(nn.Module):
    """Simple CNN for medical image classification."""

    def __init__(self, num_classes=2, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_synthetic_medical_dataset(num_samples=50, image_size=(64, 64, 32)):
    """Create synthetic medical images and labels for demonstration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create directories
        img_dir = tmpdir / "images"
        lm_dir = tmpdir / "labelmaps"
        img_dir.mkdir()
        lm_dir.mkdir()

        samples = []

        for i in range(num_samples):
            # Create synthetic 3D image (e.g., CT-like)
            image = np.random.normal(0, 50, image_size) + 1000  # HU-like values
            image = image.astype(np.float32)

            # Add some structure (simulated organs/tumors)
            center = np.array(image_size) // 2
            tumor_mask = np.zeros(image_size, dtype=bool)
            tumor_center = center + np.random.randint(-10, 10, 3)
            tumor_size = np.random.randint(5, 15, 3)

            # Create spherical tumor
            x, y, z = np.ogrid[:image_size[0], :image_size[1], :image_size[2]]
            dist_from_center = np.sqrt((x - tumor_center[0])**2 +
                                     (y - tumor_center[1])**2 +
                                     (z - tumor_center[2])**2)
            tumor_mask = dist_from_center <= np.mean(tumor_size)

            # Make tumor brighter
            image[tumor_mask] += np.random.normal(200, 50)

            # Create labelmap (0=background, 1=tumor)
            labelmap = np.zeros(image_size, dtype=np.int32)
            labelmap[tumor_mask] = 1

            # Save as NIfTI
            img_sitk = sitk.GetImageFromArray(image)
            lm_sitk = sitk.GetImageFromArray(labelmap)

            img_path = img_dir / f"patient_{i:03d}.nii.gz"
            lm_path = lm_dir / f"patient_{i:03d}_lm.nii.gz"

            sitk.WriteImage(img_sitk, str(img_path))
            sitk.WriteImage(lm_sitk, str(lm_path))

            # Assign random class (0=benign, 1=malignant)
            label = np.random.choice([0, 1], p=[0.7, 0.3])  # More benign cases

            samples.append({
                "images": [str(img_path)],
                "labelmaps": [str(lm_path)],
                "label": int(label)
            })

        return samples, str(tmpdir)


def example_pytorch_training_basic():
    """Example 1: Basic PyTorch training without augmentation."""
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC PYTORCH TRAINING (No Augmentation)")
    print("="*80)

    if not DATALOADER_AVAILABLE:
        print("Skipping: pyable_dataloader not available")
        return

    # Create synthetic dataset
    samples, data_dir = create_synthetic_medical_dataset(num_samples=20)

    # Create manifest
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump({"dataset": samples}, f)

    # Create PyTorch DataLoader
    dataset = PyableDataset(
        manifest=manifest_path,
        target_size=[64, 64, 32],
        target_spacing=[1.0, 1.0, 1.0],
        return_meta=False  # Only return tensors for training
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Initialize model
    model = MedicalImageClassifier(num_classes=2, input_channels=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(3):  # Short training for demo
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch['images'], batch['label']

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(".3f")


def example_pytorch_training_with_augmentation():
    """Example 2: PyTorch training with comprehensive data augmentation."""
    print("\n" + "="*80)
    print("EXAMPLE 2: PYTORCH TRAINING WITH DATA AUGMENTATION")
    print("="*80)

    if not DATALOADER_AVAILABLE:
        print("Skipping: pyable_dataloader not available")
        return

    # Create synthetic dataset
    samples, data_dir = create_synthetic_medical_dataset(num_samples=20)

    # Create comprehensive augmentation pipeline
    augmentation_transforms = Compose([
        # Intensity augmentations
        IntensityNormalization(method='zscore'),  # Standardize intensity
        RandomBrightness(factor_range=(-0.2, 0.2), prob=0.5),
        RandomContrast(factor_range=(0.8, 1.2), prob=0.5),
        RandomGamma(gamma_range=(0.8, 1.2), prob=0.3),

        # Spatial augmentations
        RandomTranslation(
            translation_range=[[-5, 5], [-5, 5], [-3, 3]],
            prob=0.7
        ),
        RandomRotation(
            angle_range=[[-10, 10], [-10, 10], [-5, 5]],
            prob=0.6
        ),
        RandomFlip(axes=[[1], [2], [0, 1], [0, 2], [1, 2]], prob=0.5),

        # Noise and filtering
        RandomNoise(noise_type='gaussian', mean=0, std_range=(0, 0.05), prob=0.3),
        GaussianBlur(sigma_range=(0.5, 1.5), prob=0.4),

        # Advanced augmentations
        RandomElasticDeformation(
            alpha_range=(10, 30),
            sigma_range=(3, 6),
            prob=0.3
        )
    ])

    # Create manifest
    manifest_path = os.path.join(data_dir, "manifest_aug.json")
    with open(manifest_path, 'w') as f:
        json.dump({"dataset": samples}, f)

    # Create DataLoader with augmentation
    dataset = PyableDataset(
        manifest=manifest_path,
        target_size=[64, 64, 32],
        target_spacing=[1.0, 1.0, 1.0],
        transforms=augmentation_transforms,
        return_meta=False
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Initialize model
    model = MedicalImageClassifier(num_classes=2, input_channels=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with augmented data
    print("Starting training with augmentation...")
    model.train()
    for epoch in range(3):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch['images'], batch['label']

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(".3f")


# =============================================================================
# 2. PYFE FEATURE EXTRACTION
# =============================================================================

def example_pyfe_basic():
    """Example 3: Basic feature extraction with pyfe."""
    print("\n" + "="*80)
    print("EXAMPLE 3: BASIC PYFE FEATURE EXTRACTION")
    print("="*80)

    # Create synthetic data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create 3D image and labelmap
        image = np.random.normal(1000, 200, (32, 32, 16)).astype(np.float32)
        labelmap = np.zeros((32, 32, 16), dtype=np.int32)

        # Add tumor
        center = np.array([16, 16, 8])
        x, y, z = np.ogrid[:32, :32, :16]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        tumor_mask = dist <= 5
        image[tumor_mask] += 300
        labelmap[tumor_mask] = 1

        # Save files
        img_sitk = sitk.GetImageFromArray(image)
        lm_sitk = sitk.GetImageFromArray(labelmap)

        img_path = tmpdir / "test_image.nii.gz"
        lm_path = tmpdir / "test_labelmap.nii.gz"

        sitk.WriteImage(img_sitk, str(img_path))
        sitk.WriteImage(lm_sitk, str(lm_path))

        # Create pyfe manifest
        manifest = {
            "dataset": [{
                "id": "test_sample",
                "data": [{
                    "image": str(img_path),
                    "labelmap": str(lm_path),
                    "labelmapvalue": 1,
                    "groups": [
                        {"type": "SS", "name": "SS", "options": {}},
                        {"type": "FOS", "name": "FOS", "options": {}},
                        {"type": "GLCM", "name": "GLCM", "options": {}}
                    ],
                    "groupPrefix": "tumor"
                }]
            }]
        }

        # Extract features
        print("Extracting features...")
        results, ids = exrtactMyFeatures(manifest, dimension=3, parallel=False)

        print(f"Extracted features for {len(results)} samples")
        print(f"Feature types: {list(results[0].keys()) if results else 'None'}")

        if results:
            for feature_type, features in results[0].items():
                print(f"{feature_type}: {len(features)} features")


def example_pyfe_with_dataloader():
    """Example 4: Feature extraction with dataloader integration."""
    print("\n" + "="*80)
    print("EXAMPLE 4: PYFE WITH DATALOADER INTEGRATION")
    print("="*80)

    if not DATALOADER_AVAILABLE:
        print("Skipping: pyable_dataloader not available")
        return

    # Create synthetic dataset
    samples, data_dir = create_synthetic_medical_dataset(num_samples=5)

    # Create manifest
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump({"dataset": samples}, f)

    # Create dataset with augmentation
    augmentation_transforms = Compose([
        IntensityNormalization(method='zscore'),
        RandomTranslation(translation_range=[[-3, 3], [-3, 3], [-2, 2]], prob=0.8),
        RandomFlip(axes=[1, 2], prob=0.5)
    ])

    dataset = PyableDataset(
        manifest=manifest_path,
        target_size=[32, 32, 16],
        target_spacing=[1.0, 1.0, 1.0],
        transforms=augmentation_transforms,
        return_meta=True
    )

    # Extract features from multiple augmented samples
    all_results = []

    print("Extracting features from augmented samples...")
    for idx in range(min(3, len(dataset))):  # Process first 3 samples
        sample = dataset[idx]

        # Convert tensors to temporary files
        with tempfile.TemporaryDirectory() as aug_tmpdir:
            aug_tmpdir = Path(aug_tmpdir)

            # Save augmented data
            img_array = sample['images'][0].numpy()
            lm_array = sample['labelmaps'][0].numpy()

            img_sitk = sitk.GetImageFromArray(img_array)
            lm_sitk = sitk.GetImageFromArray(lm_array.astype(np.int32))

            aug_img_path = aug_tmpdir / f"aug_{idx}_img.nii.gz"
            aug_lm_path = aug_tmpdir / f"aug_{idx}_lm.nii.gz"

            sitk.WriteImage(img_sitk, str(aug_img_path))
            sitk.WriteImage(lm_sitk, str(aug_lm_path))

            # Create pyfe manifest
            pyfe_manifest = {
                "dataset": [{
                    "id": f"aug_sample_{idx}",
                    "data": [{
                        "image": str(aug_img_path),
                        "labelmap": str(aug_lm_path),
                        "labelmapvalue": int(sample['label']),
                        "groups": [
                            {"type": "SS", "name": "SS", "options": {}},
                            {"type": "FOS", "name": "FOS", "options": {}},
                            {"type": "GLCM", "name": "GLCM", "options": {}}
                        ],
                        "groupPrefix": f"class_{sample['label']}_aug"
                    }]
                }]
            }

            # Extract features
            results, ids = exrtactMyFeatures(pyfe_manifest, dimension=3, parallel=False)
            all_results.extend(results)

    print(f"Extracted features from {len(all_results)} augmented samples")


# =============================================================================
# 3. DATA AUGMENTATION GUIDE - ALL OPTIONS
# =============================================================================

def example_augmentation_guide():
    """Example 5: Complete guide to all augmentation options."""
    print("\n" + "="*80)
    print("EXAMPLE 5: COMPLETE DATA AUGMENTATION GUIDE")
    print("="*80)

    if not DATALOADER_AVAILABLE:
        print("Skipping: pyable_dataloader not available")
        return

    print("""
DATA AUGMENTATION OPTIONS IN PYABLE-DATALOADER:

1. INTENSITY AUGMENTATIONS:
   - IntensityNormalization(method='zscore'|'minmax'|'robust')
   - RandomBrightness(factor_range=(-0.3, 0.3), prob=0.5)
   - RandomContrast(factor_range=(0.7, 1.3), prob=0.5)
   - RandomGamma(gamma_range=(0.7, 1.3), prob=0.4)
   - HistogramEqualization()
   - CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))

2. SPATIAL AUGMENTATIONS:
   - RandomTranslation(translation_range=[[-10, 10], [-10, 10], [-5, 5]], prob=0.8)
   - RandomRotation(angle_range=[[-15, 15], [-15, 15], [-10, 10]], prob=0.6)
   - RandomFlip(axes=[0, 1, 2], prob=0.5)  # Can flip any combination
   - RandomScale(scale_range=(0.9, 1.1), prob=0.4)
   - RandomAffine(scales=None, degrees=None, translation=None, prob=0.5)

3. NOISE AND FILTERING:
   - RandomNoise(noise_type='gaussian', mean=0, std_range=(0, 0.1), prob=0.3)
   - RandomNoise(noise_type='salt_pepper', amount_range=(0.01, 0.05), prob=0.2)
   - GaussianBlur(sigma_range=(0.5, 2.0), prob=0.4)
   - MedianBlur(kernel_size_range=(3, 7), prob=0.3)

4. ADVANCED AUGMENTATIONS:
   - RandomElasticDeformation(alpha_range=(20, 40), sigma_range=(5, 8), prob=0.3)
   - RandomCrop(output_size=(48, 48, 24), prob=0.6)
   - CenterCrop(output_size=(56, 56, 28))

5. COMPOSITION:
   - Compose([transform1, transform2, ...])  # Chain multiple transforms

HOW TO ASK FOR AUGMENTATION:

For PyTorch Training:
"Create a DataLoader with augmentation for medical images including:
- Intensity normalization (z-score)
- Random translations up to 5mm in each direction
- Random flips along sagittal and coronal planes
- Gaussian noise with std up to 0.05
- Elastic deformation with alpha 20-40"

For Feature Extraction:
"Extract features from augmented medical images using:
- Same augmentations as training
- Multiple samples per original image (5-10)
- Save augmented images temporarily for pyfe processing"

For Custom Scenarios:
"Apply augmentation pipeline with:
- Specific probability for each transform
- Conditional augmentations based on image characteristics
- Multi-stage augmentation (coarse → fine)"
    """)

    # Demonstrate different augmentation scenarios
    scenarios = {
        "minimal": Compose([
            IntensityNormalization(method='zscore')
        ]),

        "moderate": Compose([
            IntensityNormalization(method='zscore'),
            RandomTranslation(translation_range=[[-3, 3], [-3, 3], [-2, 2]], prob=0.6),
            RandomFlip(axes=[1, 2], prob=0.4)
        ]),

        "aggressive": Compose([
            IntensityNormalization(method='zscore'),
            RandomBrightness(factor_range=(-0.2, 0.2), prob=0.5),
            RandomContrast(factor_range=(0.8, 1.2), prob=0.5),
            RandomTranslation(translation_range=[[-5, 5], [-5, 5], [-3, 3]], prob=0.8),
            RandomRotation(angle_range=[[-10, 10], [-10, 10], [-5, 5]], prob=0.6),
            RandomFlip(axes=[0, 1, 2], prob=0.5),
            RandomNoise(noise_type='gaussian', std_range=(0, 0.03), prob=0.4),
            RandomElasticDeformation(alpha_range=(15, 30), sigma_range=(4, 7), prob=0.3)
        ]),

        "segmentation_friendly": Compose([
            IntensityNormalization(method='zscore'),
            RandomTranslation(translation_range=[[-2, 2], [-2, 2], [-1, 1]], prob=0.7),
            RandomFlip(axes=[1, 2], prob=0.5),
            RandomScale(scale_range=(0.95, 1.05), prob=0.4)
        ])
    }

    print("\nRECOMMENDED AUGMENTATION SCENARIOS:")
    for name, transforms in scenarios.items():
        print(f"\n{name.upper()}:")
        for transform in transforms.transforms:
            print(f"  - {transform.__class__.__name__}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("PYFE + PYABLE-DATALOADER COMPREHENSIVE EXAMPLES")
    print("="*80)
    print("This guide covers:")
    print("1. PyTorch training with medical images")
    print("2. Feature extraction with pyfe")
    print("3. Complete data augmentation guide")
    print("="*80)

    # Run all examples
    try:
        example_pytorch_training_basic()
        example_pytorch_training_with_augmentation()
        example_pyfe_basic()
        example_pyfe_with_dataloader()
        example_augmentation_guide()

        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()</content>
<parameter name="filePath">/home/erosm/packages/pyfe/examples/comprehensive_examples.py