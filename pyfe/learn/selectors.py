"""
Comprehensive feature selection methods for dimensionality reduction.

Includes univariate, model-based, wrapper, and dimensionality reduction methods.
"""

from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    chi2, VarianceThreshold, SelectFromModel, RFE, RFECV,
    SequentialFeatureSelector
)
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# Try to import QUBO selector (optional)
try:
    from qubo_selector import QUBOFeatureSelector
    QUBO_AVAILABLE = True
except ImportError:
    QUBO_AVAILABLE = False
    QUBOFeatureSelector = None


# ============================================================================
# SELECTOR FACTORY FUNCTIONS
# ============================================================================

def make_anova_selector(k: int = 10, mode: str = 'k_best', **kwargs):
    """ANOVA F-value based selection (classification)."""
    if mode == 'k_best':
        return SelectKBest(score_func=f_classif, k=k)
    elif mode == 'percentile':
        return SelectPercentile(score_func=f_classif, percentile=k)
    elif mode == 'fpr':
        return SelectFpr(score_func=f_classif, alpha=kwargs.get('alpha', 0.05))
    elif mode == 'fdr':
        return SelectFdr(score_func=f_classif, alpha=kwargs.get('alpha', 0.05))
    else:
        return SelectKBest(score_func=f_classif, k=k)


def make_f_regression_selector(k: int = 10, mode: str = 'k_best', **kwargs):
    """F-value based selection (regression)."""
    if mode == 'k_best':
        return SelectKBest(score_func=f_regression, k=k)
    elif mode == 'percentile':
        return SelectPercentile(score_func=f_regression, percentile=k)
    else:
        return SelectKBest(score_func=f_regression, k=k)


def make_mutual_info_selector(k: int = 10, problem_type: str = 'classification', **kwargs):
    """Mutual information based selection."""
    score_func = mutual_info_classif if problem_type == 'classification' else mutual_info_regression
    return SelectKBest(score_func=score_func, k=k)


def make_chi2_selector(k: int = 10, **kwargs):
    """Chi-squared statistic based selection (for non-negative features)."""
    return SelectKBest(score_func=chi2, k=k)


def make_variance_threshold_selector(threshold: float = 0.0, **kwargs):
    """Remove features with low variance."""
    return VarianceThreshold(threshold=threshold)


def make_rf_selector(k: int = 10, problem_type: str = 'classification', **kwargs):
    """Random Forest based feature importance selection."""
    if problem_type == 'classification':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    return SelectFromModel(estimator, max_features=k, threshold=-np.inf)


def make_extra_trees_selector(k: int = 10, problem_type: str = 'classification', **kwargs):
    """Extra Trees based feature importance selection."""
    if problem_type == 'classification':
        estimator = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        estimator = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    return SelectFromModel(estimator, max_features=k, threshold=-np.inf)


def make_l1_selector(k: int = 10, problem_type: str = 'classification', **kwargs):
    """L1-regularized linear model feature selection."""
    if problem_type == 'classification':
        estimator = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
    else:
        estimator = Lasso(alpha=0.1, random_state=42, max_iter=1000)
    return SelectFromModel(estimator, max_features=k, threshold=-np.inf)


def make_l2_selector(k: int = 10, problem_type: str = 'classification', **kwargs):
    """L2-regularized linear model feature selection."""
    if problem_type == 'classification':
        estimator = LogisticRegression(penalty='l2', solver='lbfgs', random_state=42, max_iter=1000)
    else:
        estimator = Ridge(alpha=1.0, random_state=42, max_iter=1000)
    return SelectFromModel(estimator, max_features=k, threshold=-np.inf)


def make_elasticnet_selector(k: int = 10, problem_type: str = 'classification', **kwargs):
    """ElasticNet regularized feature selection."""
    if problem_type == 'classification':
        estimator = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, 
                                      random_state=42, max_iter=1000)
    else:
        estimator = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
    return SelectFromModel(estimator, max_features=k, threshold=-np.inf)


def make_tree_selector(k: int = 10, problem_type: str = 'classification', **kwargs):
    """Decision tree based feature selection."""
    if problem_type == 'classification':
        estimator = DecisionTreeClassifier(random_state=42, max_depth=5)
    else:
        estimator = DecisionTreeRegressor(random_state=42, max_depth=5)
    return SelectFromModel(estimator, max_features=k, threshold=-np.inf)


def make_rfe_selector(k: int = 10, problem_type: str = 'classification', **kwargs):
    """Recursive Feature Elimination."""
    if problem_type == 'classification':
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    return RFE(estimator, n_features_to_select=k, step=kwargs.get('step', 1))


def make_sfs_selector(k: int = 10, problem_type: str = 'classification', direction: str = 'forward', **kwargs):
    """Sequential Feature Selection (forward or backward)."""
    if problem_type == 'classification':
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    return SequentialFeatureSelector(estimator, n_features_to_select=k, direction=direction, 
                                     cv=3, n_jobs=-1)


def make_pca_selector(k: int = 10, **kwargs):
    """Principal Component Analysis dimensionality reduction."""
    return PCA(n_components=k, random_state=42)


def make_ica_selector(k: int = 10, **kwargs):
    """Independent Component Analysis dimensionality reduction."""
    return FastICA(n_components=k, random_state=42, max_iter=1000)


def make_nmf_selector(k: int = 10, **kwargs):
    """Non-Negative Matrix Factorization dimensionality reduction."""
    return NMF(n_components=k, random_state=42, max_iter=1000)


def make_svd_selector(k: int = 10, **kwargs):
    """Truncated SVD dimensionality reduction."""
    return TruncatedSVD(n_components=k, random_state=42)


def make_lda_selector(k: Optional[int] = None, **kwargs):
    """Linear Discriminant Analysis dimensionality reduction."""
    # LDA max components = min(n_classes - 1, n_features)
    return LinearDiscriminantAnalysis(n_components=k)


def make_factor_analysis_selector(k: int = 10, **kwargs):
    """Factor Analysis dimensionality reduction."""
    return FactorAnalysis(n_components=k, random_state=42, max_iter=1000)


# QUBO-based selectors (if available)
def make_qubo_selector(k: int = 10, solver: str = 'tabu', penalty: str = 'balanced', 
                       bootstrap_samples: int = 1, **kwargs):
    """
    QUBO-based feature selection using quantum-inspired optimization.
    
    Args:
        k: Number of features to select
        solver: 'tabu', 'sa' (simulated annealing), 'sqa' (simulated quantum annealing)
        penalty: 'balanced', 'aggressive'
        bootstrap_samples: Number of bootstrap samples for stability
    """
    if not QUBO_AVAILABLE:
        raise ImportError("qubo_selector not installed. Install with: pip install qubo-selector")
    
    penalty_weight = 0.1 if penalty == 'balanced' else 1.0
    
    return QUBOFeatureSelector(
        k=k,
        solver=solver,
        penalty=penalty_weight,
        bootstrap_samples=bootstrap_samples,
        random_state=42
    )


# ============================================================================
# SELECTOR REGISTRY
# ============================================================================

SELECTOR_REGISTRY: Dict[str, Callable] = {
    # Univariate - Filter methods
    "anova": make_anova_selector,
    "f_regression": make_f_regression_selector,
    "mutual_info": make_mutual_info_selector,
    "chi2": make_chi2_selector,
    "variance_threshold": make_variance_threshold_selector,
    
    # Model-based - Embedded methods
    "random_forest": make_rf_selector,
    "extra_trees": make_extra_trees_selector,
    "l1": make_l1_selector,
    "l2": make_l2_selector,
    "elasticnet": make_elasticnet_selector,
    "decision_tree": make_tree_selector,
    
    # Wrapper methods
    "rfe": make_rfe_selector,
    "sfs_forward": lambda k, **kw: make_sfs_selector(k, direction='forward', **kw),
    "sfs_backward": lambda k, **kw: make_sfs_selector(k, direction='backward', **kw),
    
    # Dimensionality Reduction
    "pca": make_pca_selector,
    "ica": make_ica_selector,
    "nmf": make_nmf_selector,
    "svd": make_svd_selector,
    "lda": make_lda_selector,
    "factor_analysis": make_factor_analysis_selector,
}

# Add QUBO selectors if available
if QUBO_AVAILABLE:
    SELECTOR_REGISTRY.update({
        "qubo_tabu": lambda k, **kw: make_qubo_selector(k, solver='tabu', **kw),
        "qubo_sa": lambda k, **kw: make_qubo_selector(k, solver='sa', **kw),
        "qubo_sqa": lambda k, **kw: make_qubo_selector(k, solver='sqa', **kw),
    })


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_selectors() -> Dict[str, Callable]:
    """Get all available feature selectors."""
    return SELECTOR_REGISTRY.copy()


def get_selector_by_name(name: str, k: int = 10, **kwargs) -> BaseEstimator:
    """
    Get a feature selector by name.
    
    Args:
        name: Name of the selector
        k: Number of features to select
        **kwargs: Additional arguments for the selector
        
    Returns:
        Selector instance
        
    Raises:
        KeyError: If selector name not found
    """
    if name not in SELECTOR_REGISTRY:
        available = ", ".join(SELECTOR_REGISTRY.keys())
        raise KeyError(
            f"Selector '{name}' not found. Available: {available}"
        )
    return SELECTOR_REGISTRY[name](k=k, **kwargs)


def list_selectors() -> List[str]:
    """List all available selector names."""
    return sorted(SELECTOR_REGISTRY.keys())


def list_selectors_by_category() -> Dict[str, List[str]]:
    """List selectors grouped by category."""
    categories = {
        "Univariate (Filter)": [
            "anova", "f_regression", "mutual_info", "chi2", "variance_threshold"
        ],
        "Model-based (Embedded)": [
            "random_forest", "extra_trees", "l1", "l2", "elasticnet", "decision_tree"
        ],
        "Wrapper": [
            "rfe", "sfs_forward", "sfs_backward"
        ],
        "Dimensionality Reduction": [
            "pca", "ica", "nmf", "svd", "lda", "factor_analysis"
        ],
    }
    
    if QUBO_AVAILABLE:
        categories["QUBO-based (Quantum-inspired)"] = [
            "qubo_tabu", "qubo_sa", "qubo_sqa"
        ]
    
    # Filter only available selectors
    result = {}
    for category, selectors in categories.items():
        available = [s for s in selectors if s in SELECTOR_REGISTRY]
        if available:
            result[category] = available
    
    return result


def print_available_selectors():
    """Print all available selectors grouped by category."""
    categories = list_selectors_by_category()
    
    print("=" * 80)
    print(f"AVAILABLE FEATURE SELECTORS ({len(SELECTOR_REGISTRY)} total)")
    print("=" * 80)
    
    for category, selectors in categories.items():
        print(f"\n{category}:")
        for selector in selectors:
            print(f"  • {selector}")
    
    print("\n" + "=" * 80)
    if QUBO_AVAILABLE:
        print("✓ QUBO selectors available (qubo-selector installed)")
    else:
        print("✗ QUBO selectors not available (pip install qubo-selector)")
    print("=" * 80)


if __name__ == "__main__":
    print_available_selectors()
