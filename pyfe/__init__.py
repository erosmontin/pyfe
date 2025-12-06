"""
pyfe - Python Feature Extraction and Machine Learning

Feature extraction for radiomics and comprehensive ML pipeline.
"""

# Version
__version__ = "3.0.0"

# Core feature extraction (existing functionality)
from .pyfe import (
    FE, BD2DecideFE, SS, FOS, TEXTURES, GLCM, GLRLM, BenfordFE,
    exrtactMyFeatures, exrtactMyFeaturesToPandas, exrtactMyFeaturesToSQLlite
)

# Import PYRAD if available
try:
    from .pyfe import PYRAD
    PYRAD_AVAILABLE = True
except ImportError:
    PYRAD = None
    PYRAD_AVAILABLE = False

# Adapter utilities
try:
    from .pyfe_adapter import convert_manifest_to_pyfe
    ADAPTER_AVAILABLE = True
except ImportError:
    convert_manifest_to_pyfe = None
    ADAPTER_AVAILABLE = False

# Machine learning module (new)
try:
    from . import learn
    LEARN_AVAILABLE = True
except ImportError:
    LEARN_AVAILABLE = False
    learn = None

__all__ = [
    # Feature Extraction
    'FE', 'BD2DecideFE', 'SS', 'FOS', 'TEXTURES', 'GLCM', 'GLRLM', 
    'BenfordFE',
    'exrtactMyFeatures', 'exrtactMyFeaturesToPandas', 'exrtactMyFeaturesToSQLlite',
]

# Add PYRAD if available
if PYRAD_AVAILABLE:
    __all__.append('PYRAD')

__all__.extend([
    # Adapter utilities
    'convert_manifest_to_pyfe',
    
    # ML module
    'learn',
])
