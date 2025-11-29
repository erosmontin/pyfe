# pyfe.learn — LLM Integration Guide (LLM-ready)

Purpose
-------
This document is designed for two audiences:
- Human developers who want a compact, machine-readable reference to integrate `pyfe.learn` into larger pipelines.
- Large Language Models (LLMs) or automation agents that will programmatically call, combine or generate code that uses `pyfe.learn`.

Goals
-----
- Provide a small contract for inputs/outputs and error modes.
- Provide canonical code snippets for common tasks: "evaluate a pipeline", "run a grid search", "save/load experiments", "integrate with feature extractors".
- Provide JSON/YAML examples and prompt templates LLMs can use to generate or validate code.

1) Contract (short)
-------------------
- Inputs:
  - data: pandas.DataFrame or features (n_samples, n_features)
  - labels: 1D array-like (classification: categorical labels; regression: continuous)
  - problem_type: "classification" | "regression"
  - selectors: list of selector names or callables
  - estimators: list of estimator names
  - feature_counts: list of ints (k values)
  - tune: boolean (whether to run Optuna tuning for each estimator)
  - db_path, experiment_name: optional to persist results
- Outputs: JSON-like dict per run:
  - metrics: {metric_name: value, ...}
  - best_params: dict (if tuned)
  - selector_used: name
  - estimator_used: name
  - feature_count: int
  - problem_type: str
  - artifact_paths (optional): paths to saved pipeline/models
- Error modes:
  - Missing optional packages (LightGBM/XGBoost/CatBoost): registry will skip those estimators.
  - DB write failure: returns a result but logs DB error. For multi-process runs consider a stronger DB.

2) Data shapes / JSON schema (examples)
--------------------------------------
Input example (classification):
{
  "X": {
    "type": "table",
    "rows": 120,
    "cols": 200,
    "format": "pandas.DataFrame"
  },
  "y": {
    "type": "array",
    "length": 120,
    "dtype": "int or str"
  },
  "problem_type": "classification"
}

Result example (single candidate):
{
  "selector": "mutual_info",
  "feature_count": 30,
  "estimator": "random_forest",
  "tuned": true,
  "metrics": {"accuracy": 0.85, "f1_macro": 0.82},
  "best_params": {"n_estimators": 150, "max_depth": 10},
  "created_at": "2025-11-28T12:34:56Z"
}

3) Minimal code snippets (canonical)
------------------------------------
Note: these snippets assume `pyfe.learn` is installed and available in Python path.

- Run a simple grid search (programmatic):

from pyfe.learn.grid_search import GridSearchEngine

engine = GridSearchEngine(
    selectors=["mutual_info", "anova"],
    estimators=["random_forest", "svc"],
    feature_counts=[10, 30, 50],
    tune=False,
)
result = engine.run(X, y, problem_type="classification")
# result is an iterable of result dicts

- Run grid search with Optuna tuning and DB tracking:

engine = GridSearchEngine(
    selectors=["rf_importance", "anova"],
    estimators=["lightgbm", "random_forest"],
    feature_counts=[20, 50],
    tune=True,
    optuna_trials=40,
    db_path="./pyfe_experiments.db",
    experiment_name="radiomics_grid_v1"
)
for res in engine.run(X, y, problem_type="classification"):
    print(res["metrics"])  # persisted to SQLite

- Retrieve best result from DB (via `ExperimentTracker` API):

from pyfe.learn.tracker import ExperimentTracker
tracker = ExperimentTracker("./pyfe_experiments.db")
best = tracker.get_best_result("radiomics_grid_v1", metric="f1_macro", maximize=True)
print(best)

4) Integrating with feature extraction (pyfe extract -> learn)
--------------------------------------------------------------
Typical flow (pseudocode):

# 1) Extract features
features_df = extract_radiomics(files, masks)  # returns pandas.DataFrame
labels = metadata["label"].values

# 2) Preprocess (optional): scaling, imputation

# 3) Run grid search
engine = GridSearchEngine(...)
results = list(engine.run(features_df, labels, problem_type="classification"))

# 4) Choose best and retrain final pipeline on full dataset
best = choose_best(results, metric="f1_macro")
final_pipeline = build_pipeline(best)  # selector fitted on full X + estimator fitted on full X

5) API surface (quick reference)
--------------------------------
- GridSearchEngine(selectors, estimators, feature_counts, tune=False, optuna_trials=20, db_path=None, experiment_name=None, **kwargs)
  - run(X, y, problem_type)
    - yields dicts:
      {selector, feature_count, estimator, tuned, metrics, best_params}

- ExperimentTracker(db_path)
  - log_result(experiment_name, result_dict)
  - get_results(experiment_name)
  - get_best_result(experiment_name, metric, maximize=True)
  - export_to_csv(experiment_name, out_path)

- Optimizer/Optuna API
  - OptunaOptimizer(optuna_params)  # wrapper which returns best params

6) Prompt templates LLMs can use to produce code
------------------------------------------------
- Intent -> snippet prompt (few-shot):
"""
You are an assistant that writes Python code calling `pyfe.learn`.
Goal: Given a pandas DataFrame `X` and label `y` (classification), produce a short script to run a grid search over selectors ['mutual_info', 'rf_importance'], estimators ['random_forest','svc'], for k in [10,30], with Optuna tuning 30 trials and store results in './pyfe.db'. Script must:
- import necessary modules,
- create GridSearchEngine with the supplied parameters,
- run the engine and print a summary table of accuracy and f1_macro for each candidate,
- handle missing optional packages gracefully (skip if not present),
- exit with return code 0.
"""

- Higher-level orchestration prompt (compose pipelines):
"""
Create a Python function `build_and_run_pipeline(X, y, pipeline_spec)` where `pipeline_spec` is a JSON dict describing selectors, estimators, tune flag, db info. The function should validate shapes, run the grid search, return the stored best result from DB and path to a saved final pipeline if requested.
Return only the function body and imports.
"""

7) Example JSON spec for automated generation
--------------------------------------------
{
  "selectors": ["mutual_info", "rf_importance"],
  "estimators": ["random_forest", "svc"],
  "feature_counts": [10, 30],
  "tune": true,
  "optuna_trials": 30,
  "db_path": "./pyfe.db",
  "experiment_name": "auto_radiomics_v1",
  "problem_type": "classification"
}

8) Edge cases & guidance for LLMs
---------------------------------
- Always check data types: ensure `X` is a DataFrame and `y` is 1D.
- If the dataset is small and tune=true, limit `optuna_trials` to low values (10-30) to avoid overfitting.
- For imbalanced classification, prefer using `smote=True` in evaluator or add resampling steps.
- For multi-process runs that write to SQLite concurrently, prefer using a centralized DB (Postgres) or queue writes.

9) Troubleshooting quick list
-----------------------------
- Missing `lightgbm`/`xgboost`/`catboost`: the corresponding names won't appear in the estimator registry; verify with `pyfe.learn.estimators.print_available_estimators()`.
- Optuna timeouts: reduce `optuna_trials` or add time budgets per trial.
- DB locked errors: use single-process or switch DB.

10) Next steps (recommended for automation)
------------------------------------------
- Add an API to serialize full fitted pipelines to a canonical artifact store (path + metadata) and register them in `ExperimentTracker`.
- Add an HTTP microservice wrapper that can accept JSON specs (like the example above) and run grid searches asynchronously.

---
File created: `pyfe/learn/LLM_USAGE.md` — mirrors this content and is intended to be machine-friendly and human readable.
