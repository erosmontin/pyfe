"""
pyfe - Python Feature Extraction and Machine Learning

Feature extraction for radiomics and comprehensive ML pipeline.
"""

# Version
__version__ = "2.0.0"

# Core feature extraction (existing functionality)
from .pyfe import (
    FE, BD2DecideFE, SS, FOS, TEXTURES, GLCM, GLRLM, PYRAD, BenfordFE,
    exrtactMyFeatures, exrtactMyFeaturesToPandas, exrtactMyFeaturesToSQLlite
)

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
    'PYRAD', 'BenfordFE',
    'exrtactMyFeatures', 'exrtactMyFeaturesToPandas', 'exrtactMyFeaturesToSQLlite',
    
    # ML module
    'learn',
]

# Package metadata
__author__ = "Dr. Eros Montin, PhD"
__email__ = "eros.montin@gmail.com"
