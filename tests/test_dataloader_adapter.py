"""
Basic test to ensure pyfe re-exports dataloader and transforms from pyable_dataloader
"""
import pytest

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "pyfe.dataloader",
    Path(__file__).resolve().parents[1] / 'pyfe' / 'dataloader.py'
)
pyfe_dataloader = importlib.util.module_from_spec(spec)

# Provide a dummy pyable_dataloader module to allow dataloader import without
# pulling heavy dependencies (e.g., torch) during tests.
import sys
import types
fake_pyable = types.ModuleType('pyable_dataloader')
for name in [
    'PyableDataset', 'Compose', 'IntensityNormalization', 'RandomFlip', 'RandomRotation90',
    'RandomAffine', 'RandomTranslation', 'RandomRotation', 'RandomBSpline', 'RandomNoise'
]:
    setattr(fake_pyable, name, object)
sys.modules['pyable_dataloader'] = fake_pyable

spec.loader.exec_module(pyfe_dataloader)
del sys.modules['pyable_dataloader']
dataloader = pyfe_dataloader


def test_adapter_exports():
    assert hasattr(dataloader, 'PyableDataset')
    assert hasattr(dataloader, 'RandomTranslation')
    assert hasattr(dataloader, 'RandomRotation')
    assert hasattr(dataloader, 'RandomBSpline')
    # Optional convert_manifest_to_pyfe may be present if pyable_dataloader is installed
    # If not present, it should be None or absent; check attribute gracefully
    if hasattr(dataloader, 'convert_manifest_to_pyfe'):
        assert callable(dataloader.convert_manifest_to_pyfe) or dataloader.convert_manifest_to_pyfe is None
