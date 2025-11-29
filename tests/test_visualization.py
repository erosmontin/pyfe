import importlib.util
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Load the visualization module directly from the package file to avoid importing the top-level
# `pyfe` package which may have additional runtime dependencies during tests.
root = Path(__file__).resolve().parents[1]
viz_path = root / "pyfe" / "learn" / "visualization.py"
spec = importlib.util.spec_from_file_location("pyfe_learn_visualization", str(viz_path))
visualization = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(visualization)
except ImportError as e:
    pytest.skip(f"plotting dependencies not installed: {e}", allow_module_level=True)


def test_plot_grid_search_smoke():
    # small synthetic results list
    results = [
        {"selector": "mutual_info", "estimator": "random_forest", "feature_count": 10, "f1_macro": 0.8},
        {"selector": "mutual_info", "estimator": "random_forest", "feature_count": 30, "f1_macro": 0.82},
        {"selector": "anova", "estimator": "svc", "feature_count": 10, "f1_macro": 0.75},
    ]
    fig, ax = visualization.plot_grid_search_results(results, metric="f1_macro")
    assert fig is not None and ax is not None


def test_plot_feature_heatmap_smoke():
    X = np.random.randn(50, 12)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    fig, ax = visualization.plot_feature_heatmap(df, method="correlation")
    assert fig is not None and ax is not None


def test_plot_feature_matrix_smoke():
    X = np.random.randn(40, 8)
    fig, ax = visualization.plot_feature_heatmap(X, method="matrix", top_n=6)
    assert fig is not None and ax is not None


def test_plot_optuna_study_smoke():
    try:
        import optuna
    except Exception:
        # skip if optuna not installed
        return

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    fig, ax = visualization.plot_optuna_study(study)
    assert fig is not None
