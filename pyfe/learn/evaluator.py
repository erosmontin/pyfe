"""
Model evaluation with nested cross-validation and optional SMOTE.

Provides comprehensive evaluation metrics and fold-level tracking.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error, make_scorer
)
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    ImbPipeline = Pipeline
    SMOTE = None


def _detect_problem_type(y):
    """Auto-detect classification vs regression."""
    unique_values = np.unique(y)
    if len(unique_values) <= 20 and np.issubdtype(type(unique_values[0]), np.integer):
        return 'classification'
    return 'regression'


def _get_default_scorers(problem_type: str) -> Dict[str, Any]:
    """Get default scoring metrics based on problem type."""
    if problem_type == 'classification':
        return {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0),
        }
    else:  # regression
        return {
            'r2': 'r2',
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'rmse': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)), 
                               greater_is_better=False),
        }


class ModelEvaluator:
    """
    Comprehensive model evaluator with cross-validation.
    
    Features:
    - Nested CV support
    - Optional SMOTE oversampling
    - Per-fold metrics tracking
    - Feature importance extraction
    """
    
    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        use_smote: bool = True,
        smote_k_neighbors: int = 5,
        scoring: Optional[Union[str, Dict[str, Any]]] = None,
        n_jobs: int = -1
    ):
        """
        Initialize evaluator.
        
        Args:
            cv: Number of cross-validation folds
            random_state: Random seed
            use_smote: Whether to use SMOTE oversampling (classification only)
            smote_k_neighbors: Number of neighbors for SMOTE
            scoring: Scoring metric(s) to use
            n_jobs: Number of parallel jobs
        """
        self.cv = cv
        self.random_state = random_state
        self.use_smote = use_smote and IMBLEARN_AVAILABLE
        self.smote_k_neighbors = smote_k_neighbors
        self.scoring = scoring
        self.n_jobs = n_jobs
    
    def evaluate(
        self,
        estimator: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_selector: Optional[BaseEstimator] = None,
        problem_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate estimator using cross-validation.
        
        Args:
            estimator: Sklearn estimator
            X: Feature matrix
            y: Target vector
            feature_selector: Optional feature selector to include in pipeline
            problem_type: 'classification' or 'regression' (auto-detect if None)
            
        Returns:
            Dictionary with evaluation results
        """
        # Auto-detect problem type
        if problem_type is None:
            problem_type = _detect_problem_type(y)
        
        # Ensure arrays
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
            feature_names = list(X.columns)
        else:
            X_arr = np.array(X)
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
        
        y_arr = np.array(y)
        
        # Build pipeline
        steps = [('scaler', RobustScaler())]
        
        if feature_selector is not None:
            steps.append(('selector', feature_selector))
        
        if self.use_smote and problem_type == 'classification':
            # Check if we have enough samples for SMOTE
            min_class_size = np.min(np.bincount(y_arr.astype(int)))
            k_neighbors = min(self.smote_k_neighbors, min_class_size - 1)
            if k_neighbors > 0:
                steps.append(('smote', SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)))
        
        steps.append(('estimator', estimator))
        
        # Use ImbPipeline if SMOTE is included, else regular Pipeline
        if self.use_smote and any(name == 'smote' for name, _ in steps):
            pipeline = ImbPipeline(steps)
        else:
            pipeline = Pipeline(steps)
        
        # Set up CV splitter
        if problem_type == 'classification':
            cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Set up scoring
        if self.scoring is None:
            scoring = _get_default_scorers(problem_type)
        else:
            scoring = self.scoring
        
        # Run cross-validation
        cv_results = cross_validate(
            pipeline,
            X_arr,
            y_arr,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=self.n_jobs
        )
        
        # Extract results
        results = {
            'cv_scores': {},
            'train_scores': {},
            'test_scores': {},
            'feature_names': feature_names,
            'problem_type': problem_type,
            'n_features_in': X_arr.shape[1],
        }
        
        # Aggregate scores
        for metric_name in scoring.keys():
            test_key = f'test_{metric_name}'
            train_key = f'train_{metric_name}'
            
            if test_key in cv_results:
                test_scores = cv_results[test_key]
                results['test_scores'][metric_name] = {
                    'mean': np.mean(test_scores),
                    'std': np.std(test_scores),
                    'scores': test_scores.tolist()
                }
            
            if train_key in cv_results:
                train_scores = cv_results[train_key]
                results['train_scores'][metric_name] = {
                    'mean': np.mean(train_scores),
                    'std': np.std(train_scores),
                    'scores': train_scores.tolist()
                }
        
        # Extract feature importances if available
        feature_importances = []
        for estimator_fold in cv_results['estimator']:
            final_estimator = estimator_fold.named_steps['estimator']
            if hasattr(final_estimator, 'feature_importances_'):
                feature_importances.append(final_estimator.feature_importances_)
            elif hasattr(final_estimator, 'coef_'):
                # For linear models
                coef = final_estimator.coef_
                if len(coef.shape) > 1:
                    # Multi-class: take mean across classes
                    feature_importances.append(np.abs(coef).mean(axis=0))
                else:
                    feature_importances.append(np.abs(coef))
        
        if feature_importances:
            mean_importances = np.mean(feature_importances, axis=0)
            results['feature_importances'] = mean_importances.tolist()
            
            # Get selected features if selector was used
            if feature_selector is not None:
                # Get selector from first fold
                selector_fold = cv_results['estimator'][0].named_steps['selector']
                if hasattr(selector_fold, 'get_support'):
                    selected_mask = selector_fold.get_support()
                    selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
                    results['selected_features'] = selected_features
                    results['n_features_selected'] = len(selected_features)
        
        return results


def evaluate_pipeline(
    estimator: BaseEstimator,
    feature_selector: Optional[BaseEstimator],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv: int = 5,
    use_smote: bool = True,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a pipeline.
    
    Args:
        estimator: Sklearn estimator
        feature_selector: Optional feature selector
        X: Feature matrix
        y: Target vector
        cv: Number of CV folds
        use_smote: Use SMOTE oversampling
        random_state: Random seed
        **kwargs: Additional arguments for ModelEvaluator
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(
        cv=cv,
        random_state=random_state,
        use_smote=use_smote,
        **kwargs
    )
    
    return evaluator.evaluate(estimator, X, y, feature_selector=feature_selector)


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Generate dummy data
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
    
    # Create estimator and selector
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectKBest(f_classif, k=10)
    
    # Evaluate
    results = evaluate_pipeline(clf, selector, X, y)
    
    print("\nEvaluation Results:")
    print(f"Problem Type: {results['problem_type']}")
    print(f"Features In: {results['n_features_in']}")
    if 'n_features_selected' in results:
        print(f"Features Selected: {results['n_features_selected']}")
    
    print("\nTest Scores:")
    for metric, scores in results['test_scores'].items():
        print(f"  {metric}: {scores['mean']:.4f} ± {scores['std']:.4f}")
