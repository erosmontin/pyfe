"""
Quick test to verify pyfe.learn module works correctly.
"""

import numpy as np
from sklearn.datasets import make_classification

print("Testing pyfe.learn module...")
print("="*80)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from pyfe.learn import (
        get_all_classifiers,
        get_all_regressors,
        list_selectors,
        OptunaOptimizer,
        ModelEvaluator,
        GridSearchEngine,
        ExperimentTracker,
        run_full_grid_search
    )
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Check estimators
print("\n2. Testing estimators...")
try:
    classifiers = get_all_classifiers()
    regressors = get_all_regressors()
    print(f"   ✓ Found {len(classifiers)} classifiers")
    print(f"   ✓ Found {len(regressors)} regressors")
except Exception as e:
    print(f"   ✗ Estimators test failed: {e}")
    exit(1)

# Test 3: Check selectors
print("\n3. Testing selectors...")
try:
    selectors = list_selectors()
    print(f"   ✓ Found {len(selectors)} feature selectors")
except Exception as e:
    print(f"   ✗ Selectors test failed: {e}")
    exit(1)

# Test 4: Small grid search
print("\n4. Testing small grid search...")
try:
    # Generate tiny dataset
    X, y = make_classification(
        n_samples=50,
        n_features=20,
        n_informative=10,
        random_state=42
    )
    
    # Run minimal grid search
    results = run_full_grid_search(
        X, y,
        problem_type='classification',
        selectors=['anova'],
        feature_counts=[5],
        estimators=['LogisticRegression'],
        tune_hyperparams=False,  # Disable for speed
        cv=2,
        verbose=False
    )
    
    assert len(results) > 0, "No results returned"
    assert 'test_accuracy_mean' in results.columns, "Missing accuracy metric"
    
    print(f"   ✓ Grid search completed ({len(results)} results)")
    print(f"   ✓ Best accuracy: {results['test_accuracy_mean'].max():.4f}")
except Exception as e:
    print(f"   ✗ Grid search test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Optuna optimizer
print("\n5. Testing Optuna optimizer...")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    optimizer = OptunaOptimizer(
        estimator_class=RandomForestClassifier,
        n_trials=3,  # Very few for speed
        direction='maximize'
    )
    
    def objective(estimator):
        return cross_val_score(estimator, X, y, cv=2, scoring='accuracy').mean()
    
    result = optimizer.optimize(objective, show_progress_bar=False)
    
    assert 'best_params' in result, "No best params returned"
    assert 'best_value' in result, "No best value returned"
    
    print(f"   ✓ Optimization completed (best score: {result['best_value']:.4f})")
except Exception as e:
    print(f"   ✗ Optimizer test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Experiment tracker
print("\n6. Testing experiment tracker...")
try:
    import tempfile
    import os
    
    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        
        tracker = ExperimentTracker(
            db_path=db_path,
            experiment_name='test_experiment'
        )
        
        # Log a dummy result
        tracker.log_result({
            'selector': 'anova',
            'k': 10,
            'estimator': 'RandomForest',
            'problem_type': 'classification',
            'test_accuracy_mean': 0.85,
            'test_accuracy_std': 0.03,
        })
        
        # Retrieve results
        results_df = tracker.get_results()
        assert len(results_df) == 1, "Result not logged correctly"
        
        print(f"   ✓ Experiment tracking works")
except Exception as e:
    print(f"   ✗ Tracker test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nThe pyfe.learn module is working correctly.")
print("\nNext steps:")
print("  1. Run: python examples/learn_examples.py")
print("  2. See: pyfe/learn/README.md for documentation")
print("  3. Try: from pyfe.learn import run_full_grid_search")
print("="*80 + "\n")
