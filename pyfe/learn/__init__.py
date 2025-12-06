"""
pyfe.learn - Machine Learning Module for Feature Engineering and Classification/Regression

This module provides comprehensive ML capabilities including:
- Extensive classifier and regressor collections
- Multiple feature selection methods
- Smart Optuna hyperparameter optimization
- Grid search over all combinations
- Experiment tracking and comparison
"""

from .estimators import (
    get_all_classifiers,
    get_all_regressors,
    get_classifier_by_name,
    get_regressor_by_name,
    CLASSIFIER_REGISTRY,
    REGRESSOR_REGISTRY
)

from .selectors import (
    get_all_selectors,
    get_selector_by_name,
    SELECTOR_REGISTRY
)

# Import from pyml (shared utilities) - optional dependency
try:
    from pyml.tuning import (
        OptunaOptimizer,
        suggest_params
    )
    from pyml.evaluation import (
        ModelEvaluator,
    )
    from pyml.training import (
        ExperimentTracker,
    )
    PYML_AVAILABLE = True
except ImportError:
    # pyml not available - provide dummy classes
    OptunaOptimizer = None
    suggest_params = None
    ModelEvaluator = None
    ExperimentTracker = None
    PYML_AVAILABLE = False

# Local imports
from .grid_search import (
    GridSearchEngine,
    run_full_grid_search
)

# Compatibility wrapper for evaluate_pipeline (if it was used)
def evaluate_pipeline(*args, **kwargs):
    """Compatibility wrapper. Use ModelEvaluator.evaluate() instead."""
    if not PYML_AVAILABLE:
        raise ImportError("pyml package is required for ModelEvaluator. Install with: pip install pyfe[ml]")
    raise NotImplementedError(
        "evaluate_pipeline has been removed. "
        "Use ModelEvaluator from pyml.evaluation instead."
    )

def init_experiment_db(*args, **kwargs):
    """Compatibility wrapper. Use ExperimentTracker() instead."""
    if not PYML_AVAILABLE:
        raise ImportError("pyml package is required for ExperimentTracker. Install with: pip install pyfe[ml]")
    raise NotImplementedError(
        "init_experiment_db has been removed. "
        "Use ExperimentTracker from pyml.training instead."
    )

__all__ = [
    # Estimators
    'get_all_classifiers',
    'get_all_regressors',
    'get_classifier_by_name',
    'get_regressor_by_name',
    'CLASSIFIER_REGISTRY',
    'REGRESSOR_REGISTRY',
    
    # Selectors
    'get_all_selectors',
    'get_selector_by_name',
    'SELECTOR_REGISTRY',
    
    # Grid Search
    'GridSearchEngine',
    'run_full_grid_search',
    
    # Compatibility
    'evaluate_pipeline',
    'init_experiment_db',
]

# Add pyml items if available
if PYML_AVAILABLE:
    __all__.extend([
        'OptunaOptimizer',
        'suggest_params',
        'ModelEvaluator',
        'ExperimentTracker',
    ])

__version__ = "2.0.0"
