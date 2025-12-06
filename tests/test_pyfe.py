
"""
Tests for pyfe feature extraction functionality
"""
import json
import tempfile
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from pyfe import exrtactMyFeatures, exrtactMyFeaturesToPandas

# Try to import pyable dataloader and transforms
try:
    from pyfe.dataloader import (
        PyableDataset,
        Compose,
        RandomTranslation,
        RandomRotation,
        RandomFlip,
        IntensityNormalization
    )
    DATALOADER_AVAILABLE = True
except ImportError:
    DATALOADER_AVAILABLE = False
    PyableDataset = None
    Compose = None
    RandomTranslation = None
    RandomRotation = None
    RandomFlip = None
    IntensityNormalization = None


def create_synthetic_2d_image(shape=(64, 64), mean=100, std=20):
    """Create a synthetic 2D image with known statistics."""
    np.random.seed(42)  # For reproducible results
    img_array = np.random.normal(mean, std, shape).astype(np.int16)
    img_sitk = sitk.GetImageFromArray(img_array)
    img_sitk.SetSpacing([1.0, 1.0])
    img_sitk.SetOrigin([0.0, 0.0])
    return img_sitk


def create_synthetic_3d_image(shape=(32, 32, 32), mean=200, std=30):
    """Create a synthetic 3D image with known statistics."""
    np.random.seed(42)  # For reproducible results
    img_array = np.random.normal(mean, std, shape).astype(np.int16)
    img_sitk = sitk.GetImageFromArray(img_array)
    img_sitk.SetSpacing([1.0, 1.0, 1.0])
    img_sitk.SetOrigin([0.0, 0.0, 0.0])
    return img_sitk


def create_2d_labelmap(shape=(64, 64)):
    """Create a 2D labelmap with values 0, 1, 23."""
    lm_array = np.zeros(shape, dtype=np.uint8)

    # Create region with value 1
    lm_array[20:40, 20:40] = 1

    # Create region with value 23
    lm_array[10:30, 40:60] = 23

    lm_sitk = sitk.GetImageFromArray(lm_array)
    lm_sitk.SetSpacing([1.0, 1.0])
    lm_sitk.SetOrigin([0.0, 0.0])
    return lm_sitk


def create_3d_labelmap(shape=(32, 32, 32)):
    """Create a 3D labelmap with values 0, 1, 23."""
    lm_array = np.zeros(shape, dtype=np.uint8)

    # Create region with value 1 (spherical)
    center = np.array([16, 16, 16])
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                if dist < 5:
                    lm_array[x, y, z] = 1
                elif 5 <= dist < 8:
                    lm_array[x, y, z] = 23

    lm_sitk = sitk.GetImageFromArray(lm_array)
    lm_sitk.SetSpacing([1.0, 1.0, 1.0])
    lm_sitk.SetOrigin([0.0, 0.0, 0.0])
    return lm_sitk


def test_feature_extraction_2d():
    """Test feature extraction on 2D synthetic images using pyable-dataloaders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic 2D image
        img_2d = create_synthetic_2d_image()
        img_path = tmpdir / "test_2d.nii.gz"
        sitk.WriteImage(img_2d, str(img_path))

        # Create 2D labelmap
        lm_2d = create_2d_labelmap()
        lm_path = tmpdir / "test_2d_lm.nii.gz"
        sitk.WriteImage(lm_2d, str(lm_path))

        if DATALOADER_AVAILABLE:
            # Use pyable-dataloaders with augmentation (5 augmentations per image)
            print("Using PyableDataset with 5 augmentations for 2D test...")

            # Create augmentation transforms
            transforms = Compose([
                IntensityNormalization(method='zscore'),
                RandomTranslation(
                    translation_range=[[-3, 3], [-3, 3]], 
                    prob=0.9  # High probability for augmentation
                ),
                RandomRotation(
                    angle_range=[[-15, 15], [-15, 15]], 
                    prob=0.8
                ),
                RandomFlip(axes=[0, 1], prob=0.6)
            ])

            # Create manifest with 5 entries for 5 augmentations
            manifest_data = {}
            for aug_idx in range(5):  # 5 augmentations
                manifest_data[f"test_2d_aug{aug_idx}_region1"] = {
                    "images": [str(img_path)],
                    "labelmaps": [str(lm_path)],
                    "label": 1
                }
                manifest_data[f"test_2d_aug{aug_idx}_region23"] = {
                    "images": [str(img_path)],
                    "labelmaps": [str(lm_path)],
                    "label": 23
                }

            manifest_path = tmpdir / "manifest_2d.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)

            # Create PyableDataset with transforms
            dataset = PyableDataset(
                manifest=str(manifest_path),
                target_size=[64, 64],
                target_spacing=[1.0, 1.0],
                transforms=transforms,
                return_meta=True
            )

            # Extract features for all 10 augmented samples (5 aug × 2 labels)
            results = []
            ids = []

            for idx in range(len(dataset)):
                sample = dataset[idx]

                # Save augmented image and labelmap to temporary files
                aug_img_path = tmpdir / f"aug_2d_img_{idx}.nii.gz"
                aug_lm_path = tmpdir / f"aug_2d_lm_{idx}.nii.gz"
                
                # Convert tensors to SimpleITK images and save
                img_tensor = sample['images'][0]  # Assuming single image per sample
                lm_tensor = sample['labelmaps'][0]  # Assuming single labelmap per sample
                
                # Convert to numpy and then to SimpleITK
                img_array = img_tensor.numpy()
                lm_array = lm_tensor.numpy()
                
                # Create SimpleITK images (2D case)
                img_sitk = sitk.GetImageFromArray(img_array)
                lm_sitk = sitk.GetImageFromArray(lm_array.astype(np.int32))
                
                sitk.WriteImage(img_sitk, str(aug_img_path))
                sitk.WriteImage(lm_sitk, str(aug_lm_path))

                # Create pyfe manifest entry for this sample
                manifest_entry = {
                    "id": f"sample_2d_{idx}_aug",
                    "data": [{
                        "image": str(aug_img_path),
                        "labelmap": str(aug_lm_path),
                        "labelmapvalue": int(sample['label']),
                        "groups": [{"type": "SS", "name": "SS", "options": {}}, {"type": "FOS", "name": "FOS", "options": {}}],
                        "groupPrefix": f"region{int(sample['label'])}_aug"
                    }]
                }

                # Extract features
                res, id_list = exrtactMyFeatures({"dataset": [manifest_entry]}, dimension=2, parallel=False)
                results.extend(res)
                ids.extend(id_list)

        else:
            # Fallback to direct pyfe manifest
            print("Using direct pyfe manifest for 2D test...")

            # Create manifest for pyfe
            manifest = {
                "dimension": 2,
                "dataset": [
                    {
                        "id": "test_2d_region1",
                        "data": [{
                            "image": str(img_path),
                            "labelmap": str(lm_path),
                            "labelmapvalue": 1,
                            "groups": [{"type": "SS", "name": "SS", "options": {}}, {"type": "FOS", "name": "FOS", "options": {}}],
                            "groupPrefix": "region1"
                        }]
                    },
                    {
                        "id": "test_2d_region23",
                        "data": [{
                            "image": str(img_path),
                            "labelmap": str(lm_path),
                            "labelmapvalue": 23,
                            "groups": [{"type": "SS", "name": "SS", "options": {}}, {"type": "FOS", "name": "FOS", "options": {}}],
                            "groupPrefix": "region23"
                        }]
                    }
                ]
            }

            # Extract features
            results, ids = exrtactMyFeatures(manifest, dimension=2, parallel=False)

        print(f"Got {len(results)} results and {len(ids)} ids")

        # Verify results
        assert len(results) > 0, "No features extracted"
        assert len(ids) == len(results), "IDs and results length mismatch"

        # Should have features for both label values (1 and 23)
        # Each data entry should produce features
        assert len(results) >= 2, f"Expected at least 2 feature sets, got {len(results)}"

        print(f"2D test extracted {len(results)} feature sets with IDs: {ids}")


def test_feature_extraction_3d():
    """Test feature extraction on 3D synthetic images using pyable-dataloaders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic 3D image
        img_3d = create_synthetic_3d_image()
        img_path = tmpdir / "test_3d.nii.gz"
        sitk.WriteImage(img_3d, str(img_path))

        # Create 3D labelmap
        lm_3d = create_3d_labelmap()
        lm_path = tmpdir / "test_3d_lm.nii.gz"
        sitk.WriteImage(lm_3d, str(lm_path))

        if DATALOADER_AVAILABLE:
            # Use pyable-dataloaders with augmentation
            print("Using PyableDataset with augmentation for 3D test...")

            # Create augmentation transforms
            transforms = Compose([
                IntensityNormalization(method='zscore'),
                RandomTranslation(
                    translation_range=[[-2, 2], [-2, 2], [-1, 1]], 
                    prob=0.9  # High probability for augmentation
                ),
                RandomRotation(
                    angle_range=[[-10, 10], [-10, 10], [-5, 5]], 
                    prob=0.8
                ),
                RandomFlip(axes=[1, 2], prob=0.6)
            ])

            # Create manifest with 5 entries for 5 augmentations
            manifest_data = {}
            for aug_idx in range(5):  # 5 augmentations
                manifest_data[f"test_3d_aug{aug_idx}_region1"] = {
                    "images": [str(img_path)],
                    "labelmaps": [str(lm_path)],
                    "label": 1
                }
                manifest_data[f"test_3d_aug{aug_idx}_region23"] = {
                    "images": [str(img_path)],
                    "labelmaps": [str(lm_path)],
                    "label": 23
                }

            manifest_path = tmpdir / "manifest_3d.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)

            # Create PyableDataset with transforms
            dataset = PyableDataset(
                manifest=str(manifest_path),
                target_size=[32, 32, 32],
                target_spacing=[1.0, 1.0, 1.0],
                transforms=transforms,
                return_meta=True
            )

            # Extract features manually for each sample
            results = []
            ids = []

            for idx in range(len(dataset)):
                sample = dataset[idx]

                # Save augmented image and labelmap to temporary files
                aug_img_path = tmpdir / f"aug_img_{idx}.nii.gz"
                aug_lm_path = tmpdir / f"aug_lm_{idx}.nii.gz"
                
                # Convert tensors to SimpleITK images and save
                img_tensor = sample['images'][0]  # Assuming single image per sample
                lm_tensor = sample['labelmaps'][0]  # Assuming single labelmap per sample
                
                # Convert to numpy and then to SimpleITK
                img_array = img_tensor.numpy()
                lm_array = lm_tensor.numpy()
                
                # Create SimpleITK images
                img_sitk = sitk.GetImageFromArray(img_array)
                lm_sitk = sitk.GetImageFromArray(lm_array.astype(np.int32))
                
                sitk.WriteImage(img_sitk, str(aug_img_path))
                sitk.WriteImage(lm_sitk, str(aug_lm_path))

                # Create pyfe manifest entry for this sample
                manifest_entry = {
                    "id": f"sample_{idx}_aug",
                    "data": [{
                        "image": str(aug_img_path),
                        "labelmap": str(aug_lm_path),
                        "labelmapvalue": int(sample['label']),
                        "groups": [{"type": "SS", "name": "SS", "options": {}}, {"type": "FOS", "name": "FOS", "options": {}}],
                        "groupPrefix": f"region{int(sample['label'])}_aug"
                    }]
                }

                # Extract features
                res, id_list = exrtactMyFeatures({"dataset": [manifest_entry]}, dimension=3, parallel=False)
                results.extend(res)
                ids.extend(id_list)

        else:
            # Fallback to direct pyfe manifest
            print("PyableDataset not available, using direct pyfe manifest...")

            # Create manifest for pyfe
            manifest = {
                "dimension": 3,
                "dataset": [
                    {
                        "id": "test_3d_region1",
                        "data": [{
                            "image": str(img_path),
                            "labelmap": str(lm_path),
                            "labelmapvalue": 1,
                            "groups": [{"type": "SS", "name": "SS", "options": {}}, {"type": "FOS", "name": "FOS", "options": {}}],
                            "groupPrefix": "region1"
                        }]
                    },
                    {
                        "id": "test_3d_region23",
                        "data": [{
                            "image": str(img_path),
                            "labelmap": str(lm_path),
                            "labelmapvalue": 23,
                            "groups": [{"type": "SS", "name": "SS", "options": {}}, {"type": "FOS", "name": "FOS", "options": {}}],
                            "groupPrefix": "region23"
                        }]
                    }
                ]
            }

            # Extract features
            results, ids = exrtactMyFeatures(manifest, dimension=3, parallel=False)

        # Verify results
        assert len(results) > 0, "No features extracted"
        assert len(ids) == len(results), "IDs and results length mismatch"

        # Should have features for both label values (1 and 23)
        # Each data entry should produce features
        assert len(results) >= 2, f"Expected at least 2 feature sets, got {len(results)}"
        
        print(f"3D test extracted {len(results)} feature sets with IDs: {ids}")


def test_feature_extraction_to_pandas():
    """Test feature extraction with pandas output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic 2D image
        img_2d = create_synthetic_2d_image()
        img_path = tmpdir / "test_2d_pandas.nii.gz"
        sitk.WriteImage(img_2d, str(img_path))

        # Create 2D labelmap
        lm_2d = create_2d_labelmap()
        lm_path = tmpdir / "test_2d_pandas_lm.nii.gz"
        sitk.WriteImage(lm_2d, str(lm_path))

        # Try to use pyradiomics, fall back to benford if not available
        try:
            import pyradiomics
            feature_type = {"type": "pyrad", "name": "PYRAD", "options": {}}
        except ImportError:
            feature_type = {"type": "benford", "name": "BENFORD", "options": {"min": 0, "max": 5000, "bin": 128}}

        # Create manifest
        manifest = {
            "dimension": 2,
            "dataset": [{
                "id": "test_2d_pandas",
                "data": [{
                    "image": str(img_path),
                    "labelmap": str(lm_path),
                    "labelmapvalue": 1,
                    "groups": [feature_type],
                    "groupPrefix": "test"
                }]
            }]
        }

        # Extract features to pandas
        df = exrtactMyFeaturesToPandas(manifest, dimension=2, parallel=False)

        # Verify results
        assert not df.empty, "No features extracted to DataFrame"
        assert len(df.columns) > 0, "No feature columns in DataFrame"

        print(f"Pandas test extracted DataFrame with shape {df.shape}")


if __name__ == "__main__":
    test_feature_extraction_2d()
    test_feature_extraction_3d()
    test_feature_extraction_to_pandas()
    print("All tests passed!")