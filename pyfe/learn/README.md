# pyfe.learn - Machine Learning Module

Comprehensive machine learning module for feature selection, model training, and hyperparameter optimization.

## Features

### 🤖 Extensive Estimator Collection
- **40+ Classifiers**: From simple (Logistic Regression, Naive Bayes) to advanced (LightGBM, XGBoost, CatBoost)
- **35+ Regressors**: Linear models, tree ensembles, SVR, neural networks, and more
- Automatic handling of optional dependencies (LightGBM, XGBoost, CatBoost)

### 🎯 Feature Selection Methods
- **Univariate**: ANOVA, F-regression, Mutual Information, Chi2
- **Model-based**: Random Forest, Extra Trees, L1/L2, ElasticNet, Decision Trees
- **Wrapper**: RFE, Sequential Feature Selection (forward/backward)
- **Dimensionality Reduction**: PCA, ICA, NMF, SVD, LDA, Factor Analysis
- **QUBO-based** (optional): Quantum-inspired optimization (Tabu, SA, SQA)

### ⚡ Smart Hyperparameter Optimization
- Powered by **Optuna** (Tree-structured Parzen Estimator)
- Pre-configured search spaces for all estimators
- Automatic hyperparameter tuning with pruning
- Reproducible results with random seed control

### 🔍 Comprehensive Grid Search
- Search over **all combinations** of:
  - Feature selectors (20+ methods)
  - Feature counts (customizable k values)
  - Estimators (75+ classifiers + regressors)
  - Hyperparameters (via Optuna)
- Parallel execution with progress tracking
- Cross-validation with optional SMOTE oversampling

### 📊 Experiment Tracking
- SQLite database for persistent storage
- Track experiments, models, metrics, and features
- Query and compare results across experiments
- Export results to CSV/DataFrame

## Installation

```bash
# Basic installation
pip install -e .

# With advanced boosters
pip install -e ".[advanced]"

# With QUBO selectors
pip install -e ".[qubo]"

# With everything
pip install -e ".[all]"
```

## Quick Start

### Simple Example

```python
from pyfe.learn import run_full_grid_search
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=200, n_features=50, random_state=42)

# Run comprehensive grid search
results = run_full_grid_search(
    X, y,
    problem_type='classification',
    selectors=['anova', 'mutual_info', 'random_forest'],
    feature_counts=[10, 20, 50],
    estimators=['RandomForest', 'LogisticRegression', 'LightGBM'],
    tune_hyperparams=True,
    n_trials=50,
    cv=5,
    verbose=True
)

# View top results
print(results.nlargest(10, 'test_accuracy_mean'))

# Get best pipeline
best_idx = results['test_accuracy_mean'].idxmax()
best = results.loc[best_idx]
print(f"Best: {best['selector']} + {best['estimator']} (k={best['k']})")
print(f"Accuracy: {best['test_accuracy_mean']:.4f} ± {best['test_accuracy_std']:.4f}")
```

### Advanced Example with Tracking

```python
from pyfe.learn import GridSearchEngine, ExperimentTracker

# Create engine
engine = GridSearchEngine(
    problem_type='classification',
    selectors=['anova', 'pca', 'random_forest', 'l1'],
    feature_counts=[5, 10, 20, 50],
    estimators=['RandomForest', 'GradientBoosting', 'XGBoost'],
    tune_hyperparams=True,
    n_trials=100,
    cv=5,
    use_smote=True,
    experiment_name='my_experiment',
    db_path='./experiments.db'
)

# Run grid search
results = engine.run(X, y)

# Get best pipeline
best = engine.get_best_pipeline(metric='f1')
print(best)

# Load results later
from pyfe.learn import load_experiment_results
df = load_experiment_results('./experiments.db')
```

### Regression Example

```python
from pyfe.learn import run_full_grid_search
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=200, n_features=50, random_state=42)

# Run grid search
results = run_full_grid_search(
    X, y,
    problem_type='regression',
    selectors=['f_regression', 'random_forest', 'l2'],
    feature_counts=[10, 20, 50],
    estimators=['RandomForest', 'Ridge', 'XGBoost'],
    tune_hyperparams=True,
    n_trials=50,
    cv=5
)

# View top results by R²
print(results.nlargest(10, 'test_r2_mean'))
```

## Module Structure

```
pyfe/learn/
├── __init__.py              # Main API
├── estimators.py            # Classifier & regressor registry
├── selectors.py             # Feature selection methods
├── optimizer.py             # Optuna hyperparameter optimization
├── evaluator.py             # Cross-validation evaluation
├── grid_search.py           # Grid search engine
└── tracker.py               # Experiment tracking
```

## API Reference

### Core Functions

#### `run_full_grid_search(X, y, problem_type='classification', **kwargs)`
Convenience function for comprehensive grid search.

**Args:**
- `X`: Feature matrix (numpy array or pandas DataFrame)
- `y`: Target vector (numpy array or pandas Series)
- `problem_type`: 'classification' or 'regression'
- `selectors`: List of selector names (default: all fast selectors)
- `feature_counts`: List of k values (default: [5,10,20,50,100])
- `estimators`: List of estimator names (default: all available)
- `tune_hyperparams`: Use Optuna optimization (default: True)
- `n_trials`: Number of Optuna trials (default: 50)
- `cv`: Cross-validation folds (default: 5)
- `use_smote`: Use SMOTE oversampling for classification (default: True)
- `experiment_name`: Name for tracking (default: None)
- `db_path`: Database path for tracking (default: None)

**Returns:**
- DataFrame with results for all combinations

### Classes

#### `GridSearchEngine`
Main engine for grid search.

```python
engine = GridSearchEngine(
    problem_type='classification',
    selectors=['anova', 'rf'],
    feature_counts=[10, 20],
    estimators=['RandomForest', 'LogisticRegression'],
    tune_hyperparams=True,
    n_trials=50,
    cv=5,
    verbose=True
)
results = engine.run(X, y)
best = engine.get_best_pipeline(metric='accuracy')
```

#### `ModelEvaluator`
Cross-validation evaluator with SMOTE support.

```python
from pyfe.learn import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

evaluator = ModelEvaluator(cv=5, use_smote=True)
clf = RandomForestClassifier()
selector = SelectKBest(f_classif, k=10)

results = evaluator.evaluate(clf, X, y, feature_selector=selector)
```

#### `OptunaOptimizer`
Hyperparameter optimization with Optuna.

```python
from pyfe.learn import OptunaOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

optimizer = OptunaOptimizer(
    estimator_class=RandomForestClassifier,
    n_trials=100
)

def objective(estimator):
    return cross_val_score(estimator, X, y, cv=3).mean()

result = optimizer.optimize(objective)
best_clf = optimizer.get_best_estimator()
```

#### `ExperimentTracker`
Track experiments in SQLite database.

```python
from pyfe.learn import ExperimentTracker

tracker = ExperimentTracker(
    db_path='./experiments.db',
    experiment_name='my_experiment'
)

# Log results
tracker.log_result(result_dict)

# Get results
df = tracker.get_results()
best = tracker.get_best_result(metric='test_accuracy_mean')

# Export
tracker.export_to_csv('results.csv')
```

### Helper Functions

#### `get_all_classifiers()` / `get_all_regressors()`
Get dictionary of all available estimators.

#### `get_classifier_by_name(name)` / `get_regressor_by_name(name)`
Get estimator class by name.

#### `list_selectors()`
List all available selector names.

#### `get_selector_by_name(name, k=10, **kwargs)`
Get selector instance by name.

#### `print_available_estimators()`
Print all classifiers and regressors.

#### `print_available_selectors()`
Print all feature selectors grouped by category.

## Performance Tips

1. **Start Small**: Begin with a subset of methods to test quickly
2. **Parallel Processing**: Uses all CPU cores by default (`n_jobs=-1`)
3. **Hyperparameter Tuning**: Increase `n_trials` for better results (50-200 recommended)
4. **SMOTE**: Disable for balanced datasets to speed up training
5. **Feature Counts**: Test a range (e.g., [5, 10, 20, 50, 100])
6. **Experiment Tracking**: Use database to avoid re-running experiments

## Complete Example

See `examples/learn_examples.py` for comprehensive examples including:
- Classification tasks
- Regression tasks
- Custom grid search configurations
- Experiment tracking
- Result analysis

Run with:
```bash
python examples/learn_examples.py
```

## Metrics

### Classification
- `accuracy`: Accuracy score
- `precision`: Precision (weighted avg)
- `recall`: Recall (weighted avg)
- `f1`: F1 score (weighted avg)
- `roc_auc`: ROC AUC (if binary)

### Regression
- `r2`: R² coefficient of determination
- `mse`: Mean Squared Error (negative)
- `mae`: Mean Absolute Error (negative)
- `rmse`: Root Mean Squared Error (negative)

## Advanced Features

### Custom Scorer
```python
from sklearn.metrics import make_scorer, fbeta_score

custom_scorer = {
    'f2': make_scorer(fbeta_score, beta=2, average='weighted')
}

results = run_full_grid_search(
    X, y,
    scoring=custom_scorer,
    ...
)
```

### Multi-class Classification
Automatically handles multi-class problems with weighted averaging for metrics.

### Imbalanced Data
SMOTE oversampling is applied automatically in classification tasks (can be disabled).

## Troubleshooting

### "QUBOFeatureSelector not found"
Install optional QUBO package:
```bash
pip install qubo-selector
```

### "LightGBM/XGBoost/CatBoost not found"
Install advanced boosters:
```bash
pip install lightgbm xgboost catboost
```

### Memory Issues
- Reduce `n_trials` for hyperparameter tuning
- Use fewer `feature_counts`
- Limit number of estimators/selectors

### Slow Execution
- Disable `tune_hyperparams` for quick testing
- Reduce `cv` folds (e.g., cv=3)
- Exclude slow selectors (RFE, Sequential)
- Use `verbose=True` to monitor progress

## License

MIT License - see LICENSE file

## Citation

If you use pyfe.learn in your research, please cite:

```bibtex
@software{pyfe_learn,
  author = {Montin, Eros},
  title = {pyfe.learn: Comprehensive ML Framework for Radiomics},
  year = {2025},
  url = {https://github.com/erosmontin/pyfe}
}
```
