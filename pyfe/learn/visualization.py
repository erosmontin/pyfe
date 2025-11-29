"""
pyfe.learn.visualization

Plotting helpers for GridSearch results, Optuna studies, and feature heatmaps.

Functions:
- plot_grid_search_results(results, metric='f1_macro', ...)
- plot_optuna_study(study, ...)
- plot_feature_heatmap(X, ...)

These functions prefer `seaborn` + `matplotlib` and will raise a clear error with
instructions if those packages are missing.

They are intentionally lightweight: return (fig, ax) so callers (and LLMs)
can save or further customize the figures.
"""
from typing import Optional, Tuple, Union, Iterable

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
except Exception as e:
    missing = (
        "matplotlib, seaborn, pandas and numpy are required for plotting. "
        "Install with: pip install matplotlib seaborn pandas numpy"
    )
    raise ImportError(missing) from e


def _as_dataframe(results: Union[Iterable[dict], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()
    else:
        return pd.DataFrame(list(results))


def plot_grid_search_results(
    results: Union[Iterable[dict], pd.DataFrame],
    metric: str = "f1_macro",
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
    ax=None,
) -> Tuple[object, object]:
    """
    Plot aggregated metric values from a grid-search run.

    Args:
        results: Iterable of dicts (or a DataFrame) produced by GridSearchEngine or ExperimentTracker.
                 Each dict should contain keys: 'selector', 'estimator', 'feature_count', metric.
        metric: Metric name to plot (default: 'f1_macro').
        figsize: Figure size.
        title: Optional title.
        sort_by: if provided, one of ('estimator','selector','feature_count') used for ordering.
        ax: Optional matplotlib Axes to draw into.

    Returns:
        (fig, ax)

    The function produces a barplot of the chosen metric grouped by estimator and with
    feature_count on the x-axis. If the results do not contain expected keys an error is raised.
    """
    df = _as_dataframe(results)
    required = {"estimator", "selector", "feature_count", metric}
    if not required.issubset(df.columns):
        raise ValueError(f"results must contain columns: {required}. Found: {df.columns.tolist()}")

    # prepare
    df_plot = df.copy()
    df_plot["feature_count"] = df_plot["feature_count"].astype(str)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.barplot(data=df_plot, x="feature_count", y=metric, hue="estimator", ci=None, ax=ax)
    ax.set_xlabel("# features (k)")
    ax.set_ylabel(metric)
    if title is None:
        title = f"Grid search: {metric} by estimator and feature count"
    ax.set_title(title)
    ax.legend(title="estimator", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_optuna_study(
    study,
    figsize: Tuple[int, int] = (10, 4),
    show_param_importance: bool = False,
    ax=None,
    return_plotly: bool = False,
):
    """
    Plot an Optuna study's optimization history and (optionally) parameter importance.

    Args:
        study: an optuna.Study instance. If optuna.visualization is available this will return
               a plotly Figure for the optimization history when return_plotly=True.
        figsize: Matplotlib figure size for fallback plots.
        show_param_importance: If True and optuna is available, show param importance as a second plot.
        ax: Optional matplotlib axes for the main plot.
        return_plotly: If True and optuna.visualization is available, return the plotly Figure.

    Returns:
        If optuna.visualization available and return_plotly True: (plotly.Figure, None)
        Otherwise: (matplotlib.figure.Figure, matplotlib.axes.Axes)

    Note: This function gracefully falls back to a lightweight matplotlib plot if `optuna` or
    `optuna.visualization` is not installed.
    """
    try:
        import optuna
    except Exception:
        raise ImportError("optuna is required for plotting studies. Install with `pip install optuna`")

    # Prefer optuna's plotly visualizations when available
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        has_plotly = True
    except Exception:
        has_plotly = False

    if has_plotly and return_plotly:
        fig = plot_optimization_history(study)
        if show_param_importance:
            fig2 = plot_param_importances(study)
            return (fig, fig2)
        return (fig, None)

    # Fallback: simple matplotlib rendering of objective over trials
    df = study.trials_dataframe(attrs=("number", "value", "state"))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(df["number"], df["value"], marker="o", linestyle="-")
    ax.set_xlabel("trial")
    ax.set_ylabel("objective")
    ax.set_title("Optuna: optimization history")

    if show_param_importance and has_plotly:
        # if plotly visualizations are available, return a param-importance figure as well
        try:
            from optuna.visualization import plot_param_importances
            fig2 = plot_param_importances(study)
            return (fig, fig2)
        except Exception:
            # if anything goes wrong producing the secondary figure, continue with matplotlib fallback
            pass

    fig.tight_layout()
    return fig, ax


def plot_feature_heatmap(
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[Iterable[str]] = None,
    labels: Optional[Union[Iterable, pd.Series]] = None,
    method: str = "correlation",
    top_n: Optional[int] = None,
    cmap: str = "vlag",
    figsize: Tuple[int, int] = (12, 10),
    ax=None,
) -> Tuple[object, object]:
    """
    Plot a heatmap of features.

    Modes:
    - 'correlation' (default): shows feature-feature Pearson correlation matrix (n_features x n_features).
    - 'matrix': shows feature values across samples (n_samples x n_features). Use `top_n` to select the top-N
      features by variance if desired.

    Args:
        X: pandas DataFrame or numpy array of shape (n_samples, n_features).
        feature_names: Optional list of feature names. If X is a DataFrame this is inferred.
        labels: Optional labels used to annotate sample rows (only used by 'matrix' mode in the row colors).
        method: 'correlation' or 'matrix'.
        top_n: for 'matrix' mode, show only top-n features by variance.
        cmap: seaborn colormap.
        figsize: figure size.
        ax: optional axes.

    Returns:
        (fig, ax)
    """
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=(feature_names if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]))
    elif isinstance(X, pd.DataFrame):
        df = X.copy()
        if feature_names is not None:
            df.columns = feature_names
    else:
        raise ValueError("X must be a pandas DataFrame or numpy array")

    if method == "correlation":
        corr = df.corr()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        sns.heatmap(corr, cmap=cmap, center=0, square=True, ax=ax)
        ax.set_title("Feature correlation matrix")
        fig.tight_layout()
        return fig, ax

    elif method == "matrix":
        # optionally select top_n features by variance
        if top_n is not None and top_n < df.shape[1]:
            variances = df.var().sort_values(ascending=False)
            cols = variances.index[:top_n]
            df_plot = df[cols]
        else:
            df_plot = df

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        sns.heatmap(df_plot.T, cmap=cmap, cbar=True, ax=ax)
        ax.set_ylabel("features")
        ax.set_xlabel("samples")
        ax.set_title("Feature values (features x samples)")
        fig.tight_layout()
        return fig, ax

    else:
        raise ValueError("method must be one of 'correlation' or 'matrix'")
