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

# Import from pyml (shared utilities)
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

# Local imports
from .grid_search import (
    GridSearchEngine,
    run_full_grid_search
)

# Compatibility wrapper for evaluate_pipeline (if it was used)
def evaluate_pipeline(*args, **kwargs):
    """Compatibility wrapper. Use ModelEvaluator.evaluate() instead."""
    raise NotImplementedError(
        "evaluate_pipeline has been removed. "
        "Use ModelEvaluator from pyable_ml.evaluation instead."
    )

def init_experiment_db(*args, **kwargs):
    """Compatibility wrapper. Use ExperimentTracker() instead."""
    raise NotImplementedError(
        "init_experiment_db has been removed. "
        "Use ExperimentTracker from pyable_ml.training instead."
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
    
    # Optimizer
    'OptunaOptimizer',
    'suggest_params',
    
    # Evaluator
    'ModelEvaluator',
    'evaluate_pipeline',
    
    # Grid Search
    'GridSearchEngine',
    'run_full_grid_search',
    
    # Tracker
    'ExperimentTracker',
    'init_experiment_db',
]

__version__ = "2.0.0"
