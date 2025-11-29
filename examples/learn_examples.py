"""
Example script showing how to use pyfe.learn for comprehensive ML experiments.

This demonstrates:
1. Loading data
2. Running full grid search
3. Finding best pipelines
4. Tracking experiments
5. Analyzing results
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# Import pyfe.learn
from pyfe.learn import (
    GridSearchEngine,
    run_full_grid_search,
    ExperimentTracker,
    get_all_classifiers,
    get_all_regressors,
    list_selectors,
    print_available_estimators,
)


def example_classification():
    """Example: Classification task."""
    print("\n" + "="*80)
    print("CLASSIFICATION EXAMPLE")
    print("="*80 + "\n")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=300,
        n_features=100,
        n_informative=30,
        n_redundant=20,
        n_classes=2,
        random_state=42
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Run grid search with subset of methods (for speed)
    results = run_full_grid_search(
        X, y,
        problem_type='classification',
        selectors=['anova', 'mutual_info', 'random_forest', 'l1'],
        feature_counts=[10, 20, 50],
        estimators=['RandomForest', 'LogisticRegression', 'GradientBoosting'],
        tune_hyperparams=True,
        n_trials=20,  # Increase for better results
        cv=5,
        use_smote=True,
        experiment_name='classification_example',
        db_path='./examples_experiments.db',
        verbose=True
    )
    
    # Display top results
    print("\n" + "="*80)
    print("TOP 10 RESULTS BY ACCURACY")
    print("="*80)
    top_results = results.nlargest(10, 'test_accuracy_mean')[
        ['selector', 'k', 'estimator', 'test_accuracy_mean', 'test_accuracy_std', 
         'test_f1_mean', 'test_f1_std']
    ]
    print(top_results.to_string(index=False))
    
    # Find best pipeline
    print("\n" + "="*80)
    print("BEST PIPELINE")
    print("="*80)
    best_idx = results['test_accuracy_mean'].idxmax()
    best = results.loc[best_idx]
    print(f"Selector: {best['selector']}")
    print(f"Features: {best['k']}")
    print(f"Estimator: {best['estimator']}")
    print(f"Accuracy: {best['test_accuracy_mean']:.4f} ± {best['test_accuracy_std']:.4f}")
    print(f"F1 Score: {best['test_f1_mean']:.4f} ± {best['test_f1_std']:.4f}")
    if best['tuned']:
        print(f"Best Params: {best['params']}")
    
    return results


def example_regression():
    """Example: Regression task."""
    print("\n" + "="*80)
    print("REGRESSION EXAMPLE")
    print("="*80 + "\n")
    
    # Generate synthetic data
    X, y = make_regression(
        n_samples=300,
        n_features=100,
        n_informative=30,
        noise=10.0,
        random_state=42
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Run grid search
    results = run_full_grid_search(
        X, y,
        problem_type='regression',
        selectors=['f_regression', 'random_forest', 'l2'],
        feature_counts=[10, 20, 50],
        estimators=['RandomForest', 'Ridge', 'GradientBoosting'],
        tune_hyperparams=True,
        n_trials=20,
        cv=5,
        use_smote=False,  # Not applicable for regression
        experiment_name='regression_example',
        db_path='./examples_experiments.db',
        verbose=True
    )
    
    # Display top results
    print("\n" + "="*80)
    print("TOP 10 RESULTS BY R²")
    print("="*80)
    top_results = results.nlargest(10, 'test_r2_mean')[
        ['selector', 'k', 'estimator', 'test_r2_mean', 'test_r2_std',
         'test_rmse_mean', 'test_rmse_std']
    ]
    print(top_results.to_string(index=False))
    
    # Find best pipeline
    print("\n" + "="*80)
    print("BEST PIPELINE")
    print("="*80)
    best_idx = results['test_r2_mean'].idxmax()
    best = results.loc[best_idx]
    print(f"Selector: {best['selector']}")
    print(f"Features: {best['k']}")
    print(f"Estimator: {best['estimator']}")
    print(f"R²: {best['test_r2_mean']:.4f} ± {best['test_r2_std']:.4f}")
    print(f"RMSE: {-best['test_rmse_mean']:.4f} ± {best['test_rmse_std']:.4f}")
    if best['tuned']:
        print(f"Best Params: {best['params']}")
    
    return results


def example_custom_grid():
    """Example: Custom grid search with specific configurations."""
    print("\n" + "="*80)
    print("CUSTOM GRID SEARCH EXAMPLE")
    print("="*80 + "\n")
    
    # Generate data
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=20,
        n_classes=3,  # Multi-class
        random_state=42
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create custom engine
    engine = GridSearchEngine(
        problem_type='classification',
        selectors=['anova', 'pca', 'mutual_info'],
        feature_counts=[5, 10, 15],
        estimators=['RandomForest', 'ExtraTrees', 'LightGBM'] if 'LightGBM' in get_all_classifiers() else ['RandomForest', 'ExtraTrees'],
        tune_hyperparams=True,
        n_trials=15,
        cv=3,
        use_smote=True,
        experiment_name='custom_multiclass',
        db_path='./examples_experiments.db',
        verbose=True
    )
    
    # Run
    results = engine.run(X, y)
    
    # Get best pipeline
    best_pipeline = engine.get_best_pipeline(metric='accuracy')
    
    print("\n" + "="*80)
    print("BEST PIPELINE DETAILS")
    print("="*80)
    for key, value in best_pipeline.items():
        if not key.startswith('train_'):  # Skip training metrics
            print(f"{key}: {value}")
    
    return results, engine


def example_tracking():
    """Example: Experiment tracking and comparison."""
    print("\n" + "="*80)
    print("EXPERIMENT TRACKING EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize tracker
    tracker = ExperimentTracker(
        db_path='./examples_experiments.db',
        experiment_name='tracking_demo'
    )
    
    # List all experiments
    print("All Experiments:")
    experiments = tracker.list_experiments()
    print(experiments)
    
    # Get results from specific experiment
    if not experiments.empty:
        exp_id = experiments.iloc[0]['experiment_id']
        print(f"\nResults from experiment {exp_id}:")
        results = tracker.get_results(exp_id)
        print(results.head())
        
        # Get best result
        try:
            best = tracker.get_best_result('test_accuracy_mean', exp_id)
            print(f"\nBest result (accuracy):")
            print(f"  Selector: {best.get('selector')}")
            print(f"  Features: {best.get('feature_count')}")
            print(f"  Estimator: {best.get('estimator')}")
            print(f"  Accuracy: {best.get('test_accuracy_mean', 0):.4f}")
        except:
            print("\nCould not retrieve best result (may be regression experiment)")
        
        # Export to CSV
        tracker.export_to_csv('experiment_results.csv', exp_id)


def example_available_methods():
    """Example: List all available methods."""
    print("\n" + "="*80)
    print("AVAILABLE METHODS")
    print("="*80 + "\n")
    
    # Print all classifiers and regressors
    print_available_estimators()
    
    print("\n")
    
    # List selectors
    from pyfe.learn.selectors import print_available_selectors
    print_available_selectors()
    
    # Get specific counts
    classifiers = get_all_classifiers()
    regressors = get_all_regressors()
    selectors = list_selectors()
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total Classifiers: {len(classifiers)}")
    print(f"Total Regressors: {len(regressors)}")
    print(f"Total Selectors: {len(selectors)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Show available methods
    example_available_methods()
    
    # Run classification example
    clf_results = example_classification()
    
    # Run regression example
    reg_results = example_regression()
    
    # Run custom grid search
    custom_results, engine = example_custom_grid()
    
    # Show tracking capabilities
    example_tracking()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE!")
    print("="*80)
    print("\nResults saved to: ./examples_experiments.db")
    print("CSV exported to: ./experiment_results.csv")
    print("\nTo analyze results, use:")
    print("  from pyfe.learn import load_experiment_results")
    print("  df = load_experiment_results('./examples_experiments.db')")
    print("="*80 + "\n")
