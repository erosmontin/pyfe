"""
Comprehensive collection of classifiers and regressors for ML tasks.

Includes estimators from sklearn, LightGBM, XGBoost, CatBoost with smart defaults.
"""

from typing import Dict, List, Optional, Any, Type
from sklearn.base import BaseEstimator

# === Classification ===
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, 
    PassiveAggressiveClassifier, Perceptron
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

# === Regression ===
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor,
    RANSACRegressor, TheilSenRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression

# Optional advanced boosters
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    CatBoostRegressor = None


# ============================================================================
# CLASSIFIER REGISTRY
# ============================================================================

CLASSIFIER_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    # Linear Models
    "LogisticRegression": LogisticRegression,
    "RidgeClassifier": RidgeClassifier,
    "SGDClassifier": SGDClassifier,
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
    "Perceptron": Perceptron,
    
    # Tree-based
    "DecisionTree": DecisionTreeClassifier,
    "ExtraTree": ExtraTreeClassifier,
    
    # Ensemble - Trees
    "RandomForest": RandomForestClassifier,
    "ExtraTrees": ExtraTreesClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "HistGradientBoosting": HistGradientBoostingClassifier,
    "AdaBoost": AdaBoostClassifier,
    "Bagging": BaggingClassifier,
    
    # SVM
    "SVC": SVC,
    "LinearSVC": LinearSVC,
    "NuSVC": NuSVC,
    
    # Naive Bayes
    "GaussianNB": GaussianNB,
    "BernoulliNB": BernoulliNB,
    "MultinomialNB": MultinomialNB,
    "ComplementNB": ComplementNB,
    
    # Discriminant Analysis
    "LDA": LinearDiscriminantAnalysis,
    "QDA": QuadraticDiscriminantAnalysis,
    
    # Neighbors
    "KNN": KNeighborsClassifier,
    "RadiusNeighbors": RadiusNeighborsClassifier,
    
    # Neural Network
    "MLP": MLPClassifier,
    
    # Gaussian Process
    "GaussianProcess": GaussianProcessClassifier,
}

# Add optional boosters if available
if LIGHTGBM_AVAILABLE:
    CLASSIFIER_REGISTRY["LightGBM"] = LGBMClassifier
    
if XGBOOST_AVAILABLE:
    CLASSIFIER_REGISTRY["XGBoost"] = XGBClassifier
    
if CATBOOST_AVAILABLE:
    CLASSIFIER_REGISTRY["CatBoost"] = CatBoostClassifier


# ============================================================================
# REGRESSOR REGISTRY
# ============================================================================

REGRESSOR_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    # Linear Models
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "Lars": Lars,
    "LassoLars": LassoLars,
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit,
    "BayesianRidge": BayesianRidge,
    "ARDRegression": ARDRegression,
    "SGDRegressor": SGDRegressor,
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor,
    "HuberRegressor": HuberRegressor,
    "RANSACRegressor": RANSACRegressor,
    "TheilSenRegressor": TheilSenRegressor,
    
    # Tree-based
    "DecisionTree": DecisionTreeRegressor,
    "ExtraTree": ExtraTreeRegressor,
    
    # Ensemble - Trees
    "RandomForest": RandomForestRegressor,
    "ExtraTrees": ExtraTreesRegressor,
    "GradientBoosting": GradientBoostingRegressor,
    "HistGradientBoosting": HistGradientBoostingRegressor,
    "AdaBoost": AdaBoostRegressor,
    "Bagging": BaggingRegressor,
    
    # SVM
    "SVR": SVR,
    "LinearSVR": LinearSVR,
    "NuSVR": NuSVR,
    
    # Neighbors
    "KNN": KNeighborsRegressor,
    "RadiusNeighbors": RadiusNeighborsRegressor,
    
    # Neural Network
    "MLP": MLPRegressor,
    
    # Gaussian Process
    "GaussianProcess": GaussianProcessRegressor,
    
    # Other
    "KernelRidge": KernelRidge,
    "IsotonicRegression": IsotonicRegression,
}

# Add optional boosters if available
if LIGHTGBM_AVAILABLE:
    REGRESSOR_REGISTRY["LightGBM"] = LGBMRegressor
    
if XGBOOST_AVAILABLE:
    REGRESSOR_REGISTRY["XGBoost"] = XGBRegressor
    
if CATBOOST_AVAILABLE:
    REGRESSOR_REGISTRY["CatBoost"] = CatBoostRegressor


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_classifiers() -> Dict[str, Type[BaseEstimator]]:
    """Get all available classifiers."""
    return CLASSIFIER_REGISTRY.copy()


def get_all_regressors() -> Dict[str, Type[BaseEstimator]]:
    """Get all available regressors."""
    return REGRESSOR_REGISTRY.copy()


def get_classifier_by_name(name: str) -> Type[BaseEstimator]:
    """
    Get a classifier class by name.
    
    Args:
        name: Name of the classifier
        
    Returns:
        Classifier class
        
    Raises:
        KeyError: If classifier name not found
    """
    if name not in CLASSIFIER_REGISTRY:
        available = ", ".join(CLASSIFIER_REGISTRY.keys())
        raise KeyError(
            f"Classifier '{name}' not found. Available: {available}"
        )
    return CLASSIFIER_REGISTRY[name]


def get_regressor_by_name(name: str) -> Type[BaseEstimator]:
    """
    Get a regressor class by name.
    
    Args:
        name: Name of the regressor
        
    Returns:
        Regressor class
        
    Raises:
        KeyError: If regressor name not found
    """
    if name not in REGRESSOR_REGISTRY:
        available = ", ".join(REGRESSOR_REGISTRY.keys())
        raise KeyError(
            f"Regressor '{name}' not found. Available: {available}"
        )
    return REGRESSOR_REGISTRY[name]


def list_classifiers() -> List[str]:
    """List all available classifier names."""
    return sorted(CLASSIFIER_REGISTRY.keys())


def list_regressors() -> List[str]:
    """List all available regressor names."""
    return sorted(REGRESSOR_REGISTRY.keys())


def print_available_estimators():
    """Print all available estimators grouped by type."""
    print("=" * 80)
    print("AVAILABLE CLASSIFIERS ({} total)".format(len(CLASSIFIER_REGISTRY)))
    print("=" * 80)
    for name in sorted(CLASSIFIER_REGISTRY.keys()):
        print(f"  • {name}")
    
    print("\n" + "=" * 80)
    print("AVAILABLE REGRESSORS ({} total)".format(len(REGRESSOR_REGISTRY)))
    print("=" * 80)
    for name in sorted(REGRESSOR_REGISTRY.keys()):
        print(f"  • {name}")
    
    print("\n" + "=" * 80)
    optional_status = []
    if LIGHTGBM_AVAILABLE:
        optional_status.append("✓ LightGBM")
    else:
        optional_status.append("✗ LightGBM (pip install lightgbm)")
    
    if XGBOOST_AVAILABLE:
        optional_status.append("✓ XGBoost")
    else:
        optional_status.append("✗ XGBoost (pip install xgboost)")
    
    if CATBOOST_AVAILABLE:
        optional_status.append("✓ CatBoost")
    else:
        optional_status.append("✗ CatBoost (pip install catboost)")
    
    print("Optional Boosters:")
    for status in optional_status:
        print(f"  {status}")
    print("=" * 80)


if __name__ == "__main__":
    print_available_estimators()
