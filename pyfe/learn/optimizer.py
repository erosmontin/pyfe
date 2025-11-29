"""
Smart Optuna-based hyperparameter optimization for all estimators.

Provides intelligent parameter search spaces for each classifier/regressor.
"""

from typing import Dict, Any, Optional, Callable, Type
import optuna
from optuna.trial import Trial
from sklearn.base import BaseEstimator
import warnings

# Suppress Optuna logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


# ============================================================================
# PARAMETER SUGGESTION FUNCTIONS
# ============================================================================

def suggest_logistic_regression_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for LogisticRegression."""
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])
    
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "penalty": penalty,
        "max_iter": trial.suggest_int("max_iter", 100, 2000),
        "random_state": 42,
    }
    
    if penalty == "l1":
        params["solver"] = "liblinear"
    elif penalty == "l2":
        params["solver"] = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "sag", "saga"])
    elif penalty == "elasticnet":
        params["solver"] = "saga"
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    else:  # none
        params["solver"] = trial.suggest_categorical("solver", ["lbfgs", "sag", "saga"])
    
    return params


def suggest_random_forest_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for RandomForest (classifier or regressor)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
        "n_jobs": -1,
    }


def suggest_extra_trees_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for ExtraTrees."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
        "n_jobs": -1,
    }


def suggest_gradient_boosting_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for GradientBoosting."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42,
    }


def suggest_hist_gradient_boosting_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for HistGradientBoosting."""
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        "random_state": 42,
    }


def suggest_lightgbm_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for LightGBM."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", -1, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }


def suggest_xgboost_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
    }


def suggest_catboost_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for CatBoost."""
    return {
        "iterations": trial.suggest_int("iterations", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 2, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_state": 42,
        "verbose": False,
    }


def suggest_svc_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for SVC/SVR."""
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    
    params = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": kernel,
        "random_state": 42,
    }
    
    if kernel in ["rbf", "poly", "sigmoid"]:
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    
    # Add probability for classification
    if kwargs.get("problem_type") == "classification":
        params["probability"] = True
    
    return params


def suggest_knn_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for KNN."""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 2, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        "p": trial.suggest_int("p", 1, 3),  # for minkowski
    }


def suggest_decision_tree_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for DecisionTree."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
    }


def suggest_adaboost_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for AdaBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        "random_state": 42,
    }


def suggest_bagging_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for Bagging."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }


def suggest_mlp_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for MLPClassifier/Regressor."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f"n_units_l{i}", 10, 200) for i in range(n_layers)
    )
    
    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
        "max_iter": trial.suggest_int("max_iter", 200, 1000),
        "random_state": 42,
    }


def suggest_ridge_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for Ridge."""
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "random_state": 42,
    }


def suggest_lasso_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for Lasso."""
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 500, 2000),
        "random_state": 42,
    }


def suggest_elasticnet_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for ElasticNet."""
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "max_iter": trial.suggest_int("max_iter", 500, 2000),
        "random_state": 42,
    }


def suggest_qda_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for QuadraticDiscriminantAnalysis."""
    return {
        "reg_param": trial.suggest_float("reg_param", 0.0, 1.0),
    }


def suggest_gaussian_nb_params(trial: Trial, **kwargs) -> Dict[str, Any]:
    """Suggest hyperparameters for GaussianNB."""
    return {
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-7, log=True),
    }


# ============================================================================
# PARAMETER REGISTRY
# ============================================================================

PARAM_SUGGEST_REGISTRY: Dict[str, Callable] = {
    # Classification
    "LogisticRegression": suggest_logistic_regression_params,
    "RandomForest": suggest_random_forest_params,
    "RandomForestClassifier": suggest_random_forest_params,
    "RandomForestRegressor": suggest_random_forest_params,
    "ExtraTrees": suggest_extra_trees_params,
    "ExtraTreesClassifier": suggest_extra_trees_params,
    "ExtraTreesRegressor": suggest_extra_trees_params,
    "GradientBoosting": suggest_gradient_boosting_params,
    "GradientBoostingClassifier": suggest_gradient_boosting_params,
    "GradientBoostingRegressor": suggest_gradient_boosting_params,
    "HistGradientBoosting": suggest_hist_gradient_boosting_params,
    "HistGradientBoostingClassifier": suggest_hist_gradient_boosting_params,
    "HistGradientBoostingRegressor": suggest_hist_gradient_boosting_params,
    "LightGBM": suggest_lightgbm_params,
    "LGBMClassifier": suggest_lightgbm_params,
    "LGBMRegressor": suggest_lightgbm_params,
    "XGBoost": suggest_xgboost_params,
    "XGBClassifier": suggest_xgboost_params,
    "XGBRegressor": suggest_xgboost_params,
    "CatBoost": suggest_catboost_params,
    "CatBoostClassifier": suggest_catboost_params,
    "CatBoostRegressor": suggest_catboost_params,
    "SVC": suggest_svc_params,
    "SVR": suggest_svc_params,
    "NuSVC": suggest_svc_params,
    "NuSVR": suggest_svc_params,
    "KNN": suggest_knn_params,
    "KNeighborsClassifier": suggest_knn_params,
    "KNeighborsRegressor": suggest_knn_params,
    "DecisionTree": suggest_decision_tree_params,
    "DecisionTreeClassifier": suggest_decision_tree_params,
    "DecisionTreeRegressor": suggest_decision_tree_params,
    "AdaBoost": suggest_adaboost_params,
    "AdaBoostClassifier": suggest_adaboost_params,
    "AdaBoostRegressor": suggest_adaboost_params,
    "Bagging": suggest_bagging_params,
    "BaggingClassifier": suggest_bagging_params,
    "BaggingRegressor": suggest_bagging_params,
    "MLP": suggest_mlp_params,
    "MLPClassifier": suggest_mlp_params,
    "MLPRegressor": suggest_mlp_params,
    "Ridge": suggest_ridge_params,
    "RidgeClassifier": suggest_ridge_params,
    "Lasso": suggest_lasso_params,
    "ElasticNet": suggest_elasticnet_params,
    "QDA": suggest_qda_params,
    "QuadraticDiscriminantAnalysis": suggest_qda_params,
    "GaussianNB": suggest_gaussian_nb_params,
}


# ============================================================================
# OPTIMIZER CLASS
# ============================================================================

class OptunaOptimizer:
    """
    Smart Optuna-based hyperparameter optimizer.
    
    Automatically selects appropriate parameter spaces based on estimator type.
    """
    
    def __init__(
        self,
        estimator_class: Type[BaseEstimator],
        estimator_name: Optional[str] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        **suggest_kwargs
    ):
        """
        Initialize optimizer.
        
        Args:
            estimator_class: Sklearn estimator class
            estimator_name: Name for parameter lookup (auto-detected if None)
            n_trials: Number of optimization trials
            timeout: Time limit in seconds
            direction: 'maximize' or 'minimize'
            sampler: Optuna sampler (default: TPE)
            pruner: Optuna pruner (default: MedianPruner)
            **suggest_kwargs: Additional kwargs passed to suggest function
        """
        self.estimator_class = estimator_class
        self.estimator_name = estimator_name or estimator_class.__name__
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.suggest_kwargs = suggest_kwargs
        
        # Use TPE sampler by default (best for most cases)
        self.sampler = sampler or optuna.samplers.TPESampler(seed=42)
        self.pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
    
    def suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the estimator.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        if self.estimator_name in PARAM_SUGGEST_REGISTRY:
            return PARAM_SUGGEST_REGISTRY[self.estimator_name](trial, **self.suggest_kwargs)
        else:
            # Fallback: return empty dict (use estimator defaults)
            warnings.warn(
                f"No parameter suggestion function found for '{self.estimator_name}'. "
                f"Using estimator defaults."
            )
            return {}
    
    def create_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create estimator instance with given parameters."""
        return self.estimator_class(**params)
    
    def optimize(
        self,
        objective_func: Callable[[BaseEstimator], float],
        study_name: Optional[str] = None,
        show_progress_bar: bool = False
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_func: Function that takes estimator and returns score
            study_name: Name for the study
            show_progress_bar: Show Optuna progress bar
            
        Returns:
            Dictionary with best parameters and study results
        """
        study_name = study_name or f"{self.estimator_name}_optimization"
        
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name
        )
        
        def objective(trial):
            params = self.suggest_params(trial)
            estimator = self.create_estimator(params)
            return objective_func(estimator)
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress_bar
        )
        
        self.best_params = self.study.best_params
        
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "study": self.study
        }
    
    def get_best_estimator(self) -> BaseEstimator:
        """Get estimator with best parameters found."""
        if self.best_params is None:
            raise ValueError("No optimization run yet. Call optimize() first.")
        return self.create_estimator(self.best_params)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def suggest_params(estimator_name: str, trial: Trial, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to suggest parameters for an estimator.
    
    Args:
        estimator_name: Name of the estimator
        trial: Optuna trial object
        **kwargs: Additional kwargs for suggestion
        
    Returns:
        Dictionary of suggested parameters
    """
    if estimator_name in PARAM_SUGGEST_REGISTRY:
        return PARAM_SUGGEST_REGISTRY[estimator_name](trial, **kwargs)
    else:
        return {}


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    
    # Generate dummy data
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        estimator_class=RandomForestClassifier,
        n_trials=10,
        problem_type="classification"
    )
    
    # Define objective
    def objective(estimator):
        return cross_val_score(estimator, X, y, cv=3, scoring='accuracy').mean()
    
    # Optimize
    result = optimizer.optimize(objective, show_progress_bar=True)
    
    print("\nOptimization Results:")
    print(f"Best Score: {result['best_value']:.4f}")
    print(f"Best Params: {result['best_params']}")
