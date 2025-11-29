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

from .optimizer import (
    OptunaOptimizer,
    suggest_params
)

from .evaluator import (
    ModelEvaluator,
    evaluate_pipeline
)

from .grid_search import (
    GridSearchEngine,
    run_full_grid_search
)

from .tracker import (
    ExperimentTracker,
    init_experiment_db
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
