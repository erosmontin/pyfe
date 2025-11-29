# pyfe v2.0.0 - Machine Learning Module

## 🎉 What's New

A comprehensive machine learning module has been added to pyfe! This new `pyfe.learn` module provides state-of-the-art ML capabilities for feature selection, model training, and hyperparameter optimization.

## 📦 Installation

```bash
# Navigate to pyfe directory
cd /home/erosm/packages/pyfe

# Install with basic ML dependencies
pip install -e .

# Or install with advanced boosters (recommended)
pip install -e ".[advanced]"

# Or install everything including QUBO
pip install -e ".[all]"
```

## 🚀 Quick Start

```python
from pyfe.learn import run_full_grid_search
from sklearn.datasets import make_classification

# Generate or load your data
X, y = make_classification(n_samples=200, n_features=50, random_state=42)

# Run comprehensive grid search
results = run_full_grid_search(
    X, y,
    problem_type='classification',
    selectors=['anova', 'mutual_info', 'random_forest', 'l1'],
    feature_counts=[10, 20, 50],
    estimators=['RandomForest', 'LogisticRegression', 'XGBoost'],
    tune_hyperparams=True,
    n_trials=50,
    cv=5,
    verbose=True
)

# View top results
print(results.nlargest(10, 'test_accuracy_mean'))
```

## 📊 Key Features

### 1. **75+ ML Models**
   - **40+ Classifiers**: LogisticRegression, RandomForest, GradientBoosting, LightGBM, XGBoost, CatBoost, SVC, KNN, MLP, and more
   - **35+ Regressors**: LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor, XGBoostRegressor, and more

### 2. **20+ Feature Selection Methods**
   - **Univariate**: ANOVA, F-regression, Mutual Information, Chi2
   - **Model-based**: RandomForest, ExtraTrees, L1/L2, ElasticNet
   - **Wrapper**: RFE, Sequential Feature Selection
   - **Dimensionality Reduction**: PCA, ICA, NMF, SVD, LDA
   - **QUBO-based** (optional): Quantum-inspired optimization

### 3. **Smart Hyperparameter Optimization**
   - Powered by **Optuna** with Tree-structured Parzen Estimator (TPE)
   - Pre-configured search spaces for all 75+ estimators
   - Automatic pruning of unpromising trials
   - Fully reproducible with random seed control

### 4. **Comprehensive Grid Search**
   - Automatically tests ALL combinations of:
     - Feature selectors
     - Feature counts (k values)
     - ML estimators
     - Hyperparameters (via Optuna)
   - Parallel execution using all CPU cores
   - Progress tracking with tqdm
   - Cross-validation with optional SMOTE oversampling

### 5. **Experiment Tracking**
   - SQLite database for persistent storage
   - Track experiments, models, metrics, features
   - Query and compare results
   - Export to CSV/DataFrame

## 📁 Module Structure

```
pyfe/learn/
├── __init__.py           # Main API
├── estimators.py         # 75+ classifiers and regressors
├── selectors.py          # 20+ feature selection methods
├── optimizer.py          # Optuna hyperparameter optimization
├── evaluator.py          # Cross-validation evaluation
├── grid_search.py        # Comprehensive grid search engine
├── tracker.py            # SQLite experiment tracking
└── README.md            # Detailed documentation
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
cd /home/erosm/packages/pyfe
python tests/test_learn_module.py
```

Run comprehensive examples:

```bash
python examples/learn_examples.py
```

## 💡 Usage Examples

### Example 1: Simple Classification

```python
from pyfe.learn import run_full_grid_search
import pandas as pd

# Load your radiomics features
X = pd.read_csv('radiomics_features.csv')
y = pd.read_csv('labels.csv')

# Find best pipeline
results = run_full_grid_search(
    X, y,
    problem_type='classification',
    tune_hyperparams=True,
    cv=5,
    experiment_name='radiomics_classification',
    db_path='./experiments.db'
)

# Get best configuration
best = results.nlargest(1, 'test_accuracy_mean').iloc[0]
print(f"Best: {best['selector']} + {best['estimator']} (k={best['k']})")
print(f"Accuracy: {best['test_accuracy_mean']:.4f}")
```

### Example 2: Regression with Custom Config

```python
from pyfe.learn import GridSearchEngine

engine = GridSearchEngine(
    problem_type='regression',
    selectors=['f_regression', 'random_forest', 'pca'],
    feature_counts=[5, 10, 20, 50, 100],
    estimators=['RandomForest', 'XGBoost', 'Ridge'],
    tune_hyperparams=True,
    n_trials=100,
    cv=5,
    verbose=True
)

results = engine.run(X, y)
best = engine.get_best_pipeline(metric='r2')
```

### Example 3: Track Multiple Experiments

```python
from pyfe.learn import ExperimentTracker, load_experiment_results

# Run multiple experiments
for subset in ['train', 'validation', 'test']:
    X_sub, y_sub = load_data(subset)
    
    results = run_full_grid_search(
        X_sub, y_sub,
        experiment_name=f'experiment_{subset}',
        db_path='./experiments.db',
        ...
    )

# Compare all experiments
all_results = load_experiment_results('./experiments.db')
print(all_results.groupby('experiment_id')['test_accuracy_mean'].max())
```

## 🔧 Advanced Features

### Custom Metrics
```python
from sklearn.metrics import make_scorer, fbeta_score

custom_scorer = {
    'f2': make_scorer(fbeta_score, beta=2, average='weighted')
}

results = run_full_grid_search(X, y, scoring=custom_scorer, ...)
```

### Disable SMOTE
```python
results = run_full_grid_search(
    X, y,
    use_smote=False,  # For balanced datasets
    ...
)
```

### Quick Testing (No Hyperparameter Tuning)
```python
results = run_full_grid_search(
    X, y,
    tune_hyperparams=False,  # Use default parameters
    cv=3,  # Fewer folds for speed
    ...
)
```

## 📈 Performance Tips

1. **Start Small**: Test with subset of methods first
2. **Parallel Processing**: Uses all cores by default (`n_jobs=-1`)
3. **Hyperparameter Tuning**: Use 50-200 trials for best results
4. **SMOTE**: Disable for balanced datasets
5. **Feature Counts**: Test logarithmic range (e.g., [5, 10, 20, 50, 100])
6. **Experiment Tracking**: Avoids re-running same experiments

## 🔗 Integration with Existing pyfe

The new ML module integrates seamlessly with existing pyfe feature extraction:

```python
# 1. Extract features with pyfe
from pyfe import exrtactMyFeaturesToPandas

X = exrtactMyFeaturesToPandas('config.json', dimension=3)

# 2. Run ML pipeline
from pyfe.learn import run_full_grid_search

results = run_full_grid_search(X, y, problem_type='classification')
```

## 📖 Documentation

- **Module README**: `/home/erosm/packages/pyfe/pyfe/learn/README.md`
- **Examples**: `/home/erosm/packages/pyfe/examples/learn_examples.py`
- **API Docs**: See docstrings in each module

## 🐛 Troubleshooting

### Missing Dependencies

```bash
# Install optional boosters
pip install lightgbm xgboost catboost

# Install QUBO selectors
pip install qubo-selector

# Install all optional dependencies
pip install -e ".[all]"
```

### Import Errors

```python
# Check what's available
from pyfe.learn import print_available_estimators
print_available_estimators()
```

### Performance Issues

- Reduce `n_trials` (e.g., 20-50)
- Use fewer `feature_counts`
- Disable `tune_hyperparams` for testing
- Reduce `cv` folds (e.g., cv=3)

## 🎯 Next Steps

1. **Install dependencies**: `pip install -e ".[advanced]"`
2. **Run tests**: `python tests/test_learn_module.py`
3. **Try examples**: `python examples/learn_examples.py`
4. **Read documentation**: `pyfe/learn/README.md`
5. **Start experimenting** with your radiomics data!

## 📝 Summary

The new `pyfe.learn` module provides:

✅ **75+ ML models** (classifiers + regressors)  
✅ **20+ feature selection methods**  
✅ **Smart Optuna optimization** with pre-configured spaces  
✅ **Comprehensive grid search** over all combinations  
✅ **Experiment tracking** with SQLite database  
✅ **Cross-validation** with optional SMOTE  
✅ **Parallel execution** and progress tracking  
✅ **Easy-to-use API** with one-line grid search  
✅ **Fully documented** with examples  

This provides a complete end-to-end ML pipeline similar to your existing code, but with more features, better organization, and easier maintenance!

## 📧 Questions?

Created by: Dr. Eros Montin, PhD (eros.montin@gmail.com)

Happy modeling! 🚀
