"""
Comprehensive grid search over all combinations of:
- Feature selectors
- Feature counts
- Estimators
- Hyperparameter optimization

This is the main engine for finding optimal ML pipelines.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from itertools import product
import warnings
from tqdm import tqdm

from .estimators import get_all_classifiers, get_all_regressors, get_classifier_by_name, get_regressor_by_name
from .selectors import get_selector_by_name, list_selectors
from .optimizer import OptunaOptimizer
from .evaluator import ModelEvaluator
from .tracker import ExperimentTracker

warnings.filterwarnings('ignore')


class GridSearchEngine:
    """
    Comprehensive grid search engine for ML pipeline optimization.
    
    Searches over:
    1. Feature selectors (multiple methods)
    2. Feature counts (different k values)
    3. Estimators (classifiers or regressors)
    4. Hyperparameters (via Optuna)
    """
    
    def __init__(
        self,
        problem_type: str = 'classification',
        selectors: Optional[List[str]] = None,
        feature_counts: Optional[List[int]] = None,
        estimators: Optional[List[str]] = None,
        tune_hyperparams: bool = True,
        n_trials: int = 50,
        cv: int = 5,
        use_smote: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        experiment_name: Optional[str] = None,
        db_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize grid search engine.
        
        Args:
            problem_type: 'classification' or 'regression'
            selectors: List of selector names to try (None = all)
            feature_counts: List of k values to try (None = [5,10,20,50,100])
            estimators: List of estimator names to try (None = all)
            tune_hyperparams: Whether to use Optuna for hyperparameter tuning
            n_trials: Number of Optuna trials per estimator
            cv: Number of cross-validation folds
            use_smote: Use SMOTE oversampling (classification only)
            random_state: Random seed
            n_jobs: Number of parallel jobs
            experiment_name: Name for experiment tracking
            db_path: Path to SQLite database for tracking
            verbose: Show progress bars
        """
        self.problem_type = problem_type
        self.tune_hyperparams = tune_hyperparams
        self.n_trials = n_trials
        self.cv = cv
        self.use_smote = use_smote
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Default selectors
        if selectors is None:
            all_selectors = list_selectors()
            # Use fast selectors by default (exclude slow wrapper methods)
            self.selectors = [s for s in all_selectors if s not in ['rfe', 'sfs_forward', 'sfs_backward']]
        else:
            self.selectors = selectors
        
        # Default feature counts
        if feature_counts is None:
            self.feature_counts = [5, 10, 20, 50, 100]
        else:
            self.feature_counts = feature_counts
        
        # Default estimators
        if estimators is None:
            if problem_type == 'classification':
                est_dict = get_all_classifiers()
            else:
                est_dict = get_all_regressors()
            self.estimators = list(est_dict.keys())
        else:
            self.estimators = estimators
        
        # Setup evaluator
        self.evaluator = ModelEvaluator(
            cv=cv,
            random_state=random_state,
            use_smote=use_smote,
            n_jobs=n_jobs
        )
        
        # Setup tracker
        if db_path or experiment_name:
            self.tracker = ExperimentTracker(
                db_path=db_path,
                experiment_name=experiment_name or 'grid_search'
            )
        else:
            self.tracker = None
        
        self.results: List[Dict[str, Any]] = []
    
    def run(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        max_features: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run comprehensive grid search.
        
        Args:
            X: Feature matrix
            y: Target vector
            max_features: Maximum number of features to try (filters feature_counts)
            
        Returns:
            DataFrame with all results
        """
        # Filter feature counts based on max_features
        if max_features is not None:
            feature_counts = [k for k in self.feature_counts if k <= max_features]
        else:
            feature_counts = self.feature_counts
        
        # Get all combinations
        combinations = list(product(self.selectors, feature_counts, self.estimators))
        total = len(combinations)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Starting Grid Search")
            print(f"{'='*80}")
            print(f"Problem Type: {self.problem_type}")
            print(f"Selectors: {len(self.selectors)}")
            print(f"Feature Counts: {feature_counts}")
            print(f"Estimators: {len(self.estimators)}")
            print(f"Total Combinations: {total}")
            print(f"Hyperparameter Tuning: {self.tune_hyperparams} ({self.n_trials} trials)")
            print(f"Cross-Validation: {self.cv} folds")
            print(f"{'='*80}\n")
        
        # Progress bar
        pbar = tqdm(combinations, disable=not self.verbose, desc="Grid Search")
        
        for selector_name, k, estimator_name in pbar:
            if self.verbose:
                pbar.set_description(f"{selector_name[:15]:15s} k={k:3d} {estimator_name[:20]:20s}")
            
            try:
                result = self._evaluate_combination(
                    X, y, selector_name, k, estimator_name
                )
                self.results.append(result)
                
                # Track if tracker is available
                if self.tracker:
                    self.tracker.log_result(result)
            
            except Exception as e:
                if self.verbose:
                    print(f"\n⚠ Error with {selector_name}, k={k}, {estimator_name}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Grid Search Complete!")
            print(f"{'='*80}")
            print(f"Total Combinations Tested: {len(self.results)}/{total}")
            print(f"{'='*80}\n")
        
        return df
    
    def _evaluate_combination(
        self,
        X, y,
        selector_name: str,
        k: int,
        estimator_name: str
    ) -> Dict[str, Any]:
        """Evaluate a single combination."""
        # Get estimator class
        if self.problem_type == 'classification':
            estimator_class = get_classifier_by_name(estimator_name)
        else:
            estimator_class = get_regressor_by_name(estimator_name)
        
        # Get selector
        try:
            selector = get_selector_by_name(
                selector_name,
                k=k,
                problem_type=self.problem_type
            )
        except Exception as e:
            # Some selectors may not support all feature counts
            raise ValueError(f"Selector {selector_name} failed with k={k}: {e}")
        
        # Optimize hyperparameters if requested
        if self.tune_hyperparams:
            optimizer = OptunaOptimizer(
                estimator_class=estimator_class,
                estimator_name=estimator_name,
                n_trials=self.n_trials,
                problem_type=self.problem_type
            )
            
            # Define objective for optimization
            def objective(estimator):
                eval_result = self.evaluator.evaluate(
                    estimator, X, y,
                    feature_selector=selector,
                    problem_type=self.problem_type
                )
                # Return first test metric
                first_metric = list(eval_result['test_scores'].keys())[0]
                return eval_result['test_scores'][first_metric]['mean']
            
            opt_result = optimizer.optimize(objective, show_progress_bar=False)
            estimator = optimizer.get_best_estimator()
            best_params = opt_result['best_params']
        else:
            estimator = estimator_class()
            best_params = {}
        
        # Final evaluation
        eval_result = self.evaluator.evaluate(
            estimator, X, y,
            feature_selector=selector,
            problem_type=self.problem_type
        )
        
        # Compile result
        result = {
            'selector': selector_name,
            'k': k,
            'estimator': estimator_name,
            'problem_type': self.problem_type,
            'tuned': self.tune_hyperparams,
            'params': best_params,
            'n_features_in': eval_result['n_features_in'],
        }
        
        # Add all test scores
        for metric, scores in eval_result['test_scores'].items():
            result[f'test_{metric}_mean'] = scores['mean']
            result[f'test_{metric}_std'] = scores['std']
        
        # Add train scores
        for metric, scores in eval_result['train_scores'].items():
            result[f'train_{metric}_mean'] = scores['mean']
            result[f'train_{metric}_std'] = scores['std']
        
        # Add feature info
        if 'n_features_selected' in eval_result:
            result['n_features_selected'] = eval_result['n_features_selected']
        
        return result
    
    def get_best_pipeline(self, metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Get the best pipeline based on a metric.
        
        Args:
            metric: Metric name to optimize (e.g., 'accuracy', 'f1', 'r2')
            
        Returns:
            Dictionary with best pipeline configuration
        """
        if not self.results:
            raise ValueError("No results available. Run grid search first.")
        
        df = pd.DataFrame(self.results)
        metric_col = f'test_{metric}_mean'
        
        if metric_col not in df.columns:
            available = [c for c in df.columns if c.startswith('test_') and c.endswith('_mean')]
            raise ValueError(f"Metric '{metric}' not found. Available: {available}")
        
        best_idx = df[metric_col].idxmax()
        return df.loc[best_idx].to_dict()


def run_full_grid_search(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    problem_type: str = 'classification',
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to run full grid search.
    
    Args:
        X: Feature matrix
        y: Target vector
        problem_type: 'classification' or 'regression'
        **kwargs: Additional arguments for GridSearchEngine
        
    Returns:
        DataFrame with results
    """
    engine = GridSearchEngine(problem_type=problem_type, **kwargs)
    return engine.run(X, y)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate dummy data
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=20,
        random_state=42
    )
    
    # Run grid search (small example)
    results = run_full_grid_search(
        X, y,
        problem_type='classification',
        selectors=['anova', 'mutual_info', 'random_forest'],
        feature_counts=[5, 10, 20],
        estimators=['RandomForest', 'LogisticRegression'],
        tune_hyperparams=True,
        n_trials=10,
        cv=3
    )
    
    print("\nTop 5 Results:")
    print(results.nlargest(5, 'test_accuracy_mean')[
        ['selector', 'k', 'estimator', 'test_accuracy_mean', 'test_accuracy_std']
    ])
