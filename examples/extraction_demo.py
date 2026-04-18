#!/usr/bin/env python3
"""
examples/extraction_demo.py

Create 3 synthetic 3D images + ROI masks (NIfTI via SimpleITK), convert the
dataset to a pyfe-style manifest and run feature extraction via the package
API, saving results to SQLite.
"""

import os
import json
import sys
import numpy as np
from pathlib import Path

# 1) Import pyfe API
from pyfe import exrtactMyFeaturesToSQLlite, convert_manifest_to_pyfe, PYRAD_AVAILABLE

# 2) Optional: import SITK to write nifti images
try:
    import SimpleITK as sitk
except Exception:
    sitk = None

def make_sphere_mask(shape, center=None, radius=None):
    if center is None:
        center = (s//2 for s in shape)
    if radius is None:
        radius = min(shape) // 6
    center = tuple(center)
    z = np.arange(shape[0]) - center[0]
    y = np.arange(shape[1]) - center[1]
    x = np.arange(shape[2]) - center[2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return (zz**2 + yy**2 + xx**2) <= (radius**2)

def generate_image(shape=(64,64,32), seed=0):
    rng = np.random.RandomState(seed)
    img = rng.normal(loc=100.0, scale=20.0, size=shape).astype(np.float32)
    mask = make_sphere_mask(shape, radius=min(shape)//6)
    img[mask] += rng.uniform(30.0, 80.0)
    return img, mask.astype(np.uint8)

def save_nifti(np_arr, path):
    if sitk is None:
        raise RuntimeError("SimpleITK is required to save NIfTI; install it or adjust the script.")
    img = sitk.GetImageFromArray(np_arr)
    sitk.WriteImage(img, str(path))

def build_pyfe_manifest(dataset_dir: Path, n_subjects=3):
    """Generate NIfTI images + masks, and produce a pyfe manifest JSON file."""
    dataset = []
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for sid in range(1, n_subjects + 1):
        image_path = dataset_dir / f"image_{sid:03d}.nii.gz"
        mask_path = dataset_dir / f"mask_{sid:03d}.nii.gz"

        img, mask = generate_image(seed=sid)
        save_nifti(img, image_path)
        save_nifti(mask, mask_path)

            # Minimal non-aug config for demo; we want PYRAD and Benford features
            # Use PYRAD if available, otherwise rely on Benford
        groups = []
        if PYRAD_AVAILABLE:
            # minimal config for pyradiomics
            groups.append({"type": "pyrad", "options": {"bin": 32, "radius": 1}, "name": "PYRAD"})
            # Benford feature extraction requires no special options
            groups.append({"type": "benford", "options": {}, "name": "Benford"})

        dataset.append({
            "id": f"subj_{sid}",
            "data": [
                {
                    "image": str(image_path),
                    "labelmap": str(mask_path),
                    "labelmapvalue": 1,
                    "groups": groups,
                    "groupPrefix": "Region"
                }
            ],
            # For demonstration we won't perform aug, but pyfe supports it
        })

    # Dump manifest
    manifest = {"dimension": 3, "dataset": dataset}
    manifest_path = dataset_dir / "manifest_pyfe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path

def main():
    base = Path("data/demo_pyfe")
    base.mkdir(parents=True, exist_ok=True)

    # Generate dataset + manifest
    manifest_path = build_pyfe_manifest(base, n_subjects=3)
    print(f"Manifest created: {manifest_path}")

    # Output DB location
    db_path = base / "features.db"
    # If radiomics is available, add PYRAD conf; else FOS already set as FOS
    # We'll pass the manifest file to exrtactMyFeaturesToSQLlite
    import inspect
    kwargs = dict(
        max_level=3,
        parallel=True,
        augonly=False,
        saveimages=None,
        db=str(db_path),
        table_name="features",
        extraction_configurations=None,
        log=None
    )
    # Only pass the newer kwargs if the function signature supports them
    sig = inspect.signature(exrtactMyFeaturesToSQLlite)
    if 'reuse_pyrad_extractor' in sig.parameters:
        kwargs['reuse_pyrad_extractor'] = True
        kwargs['pyrad_settings'] = {"bin": 32, "radius": 1}
    conf = exrtactMyFeaturesToSQLlite(str(manifest_path), 3, **kwargs)
    print("Extraction finished — DB:", conf.get("db"))
    # Read DB and show a summary
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql("SELECT * FROM features LIMIT 5;", conn)
    print("Sample rows from DB:")
    print(df.head().to_string(index=False))
    conn.close()

    # --- Faster in-memory extraction demonstration (optional) ---
    # This method avoids on-disk file writes and can reuse a per-worker pyradiomics
    # extractor for faster processing. It uses SimpleITK images in memory.
    try:
        from pyfe import extract_features_from_arrays
        if sitk is not None:
            # Load back images into memory as SITK objects
            images = [sitk.ReadImage(str(base / f"image_{i:03d}.nii.gz")) for i in range(1, 4)]
            rois = [sitk.ReadImage(str(base / f"mask_{i:03d}.nii.gz")) for i in range(1, 4)]
            groups = [g["data"][0]["groups"] for g in json.loads(manifest_path.read_text())["dataset"]]
            ids = [f"subj_{i}" for i in range(1, 4)]
            features_mem, ids_mem = extract_features_from_arrays(images, rois, groups, ids=ids, parallel=False, reuse_pyrad_extractor=True, pyrad_settings={"bin": 32, "radius": 1})
            print("In-memory feature extraction (sample):")
            for i, fid in enumerate(ids_mem):
                print(fid, list(features_mem[i].keys()))
    except Exception:
        # If pyfe doesn't expose the function (older version), ignore
        pass

if __name__ == "__main__":
    main()