from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alpha_gen.core.metrics import evaluate_factor
from alpha_gen.core.preprocess import build_transform_cache, load_panel
from alpha_gen.core.gene import load_field_rules
from alpha_gen.core.utils import get_rolling_windows


# ---------------------------------------------------------------------------
# Traditional gplearn baseline.
#
# This is intentionally a baseline, not the main reproduction. gplearn learns
# arbitrary symbolic trees on flattened samples, while the alpha_gen main path
# uses constrained structured expressions and NSGA-II. The baseline is useful
# to compare against the older tree-GP research style.
#
# Run with:
#   conda run -n pytorch python alpha_gen/examples/run_gplearn_baseline.py
# ---------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mock_tmt_daily.parquet"
META_PATH = ROOT / "data" / "mock_tmt_metadata.json"
RESULT_DIR = ROOT / "results"


FEATURE_SPECS = [
    ("log_book_equity", "book_equity", True),
    # market_cap is already stored as log size in the prepared panel.
    ("log_market_cap", "market_cap", False),
    ("log_enterprise_value", "enterprise_value", True),
    ("log_revenue_ttm", "revenue_ttm", True),
    ("log_net_profit_ttm", "net_profit_ttm", True),
    ("log_operating_profit_ttm", "operating_profit_ttm", True),
    ("rating_score_30d", "rating_score_30d", False),
    ("rating_score_90d", "rating_score_90d", False),
    ("rating_score_180d", "rating_score_180d", False),
]


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize every feature by date before flattening to sklearn samples."""

    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _build_feature_frame(cache, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Create sklearn X/y from pivot matrices."""

    feature_parts = []
    for output_name, field, use_log in FEATURE_SPECS:
        matrix = cache.get_current(field, use_log=use_log).loc[dates]
        matrix = _cross_sectional_zscore(matrix)
        stacked = matrix.stack(future_stack=True).rename(output_name)
        stacked.index = stacked.index.set_names(["Datetime", "Contract"])
        feature_parts.append(stacked)

    x_df = pd.concat(feature_parts, axis=1)
    y = cache.label.loc[dates].stack(future_stack=True).rename("label_20d")
    y.index = y.index.set_names(["Datetime", "Contract"])
    tradeable = cache.tradeable.loc[dates].stack(future_stack=True).astype(bool)
    tradeable.index = tradeable.index.set_names(["Datetime", "Contract"])

    data = x_df.join(y).where(tradeable, np.nan).dropna()
    return data.drop(columns=["label_20d"]), data["label_20d"]


def _safe_import_gplearn():
    try:
        from gplearn.genetic import SymbolicTransformer
    except ImportError as exc:
        raise SystemExit(
            "gplearn is not installed. Install in the pytorch environment with:\n"
            "conda run -n pytorch pip install gplearn -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc
    return SymbolicTransformer


def _patch_gplearn_for_new_sklearn(estimator) -> None:
    """Patch gplearn 0.4.2 for newer scikit-learn if needed.

    gplearn 0.4.2 expects `_validate_data` from older sklearn BaseEstimator
    versions. Newer sklearn versions removed that helper, so a minimal method is
    attached to this estimator instance only.
    """

    if hasattr(estimator, "_validate_data"):
        return

    from sklearn.utils.validation import check_X_y

    def _validate_data(self, X, y, y_numeric=True):
        x_checked, y_checked = check_X_y(X, y, y_numeric=y_numeric, dtype=np.float64)
        self.n_features_in_ = x_checked.shape[1]
        return x_checked, y_checked

    estimator._validate_data = _validate_data.__get__(estimator, estimator.__class__)


def _program_to_factor(program_values: np.ndarray, sample_index: pd.MultiIndex) -> pd.DataFrame:
    """Convert flattened gplearn output back to Datetime x Contract matrix."""

    series = pd.Series(program_values, index=sample_index, name="gplearn_factor")
    factor = series.unstack("Contract")
    factor.index.name = "Datetime"
    factor.columns.name = "Contract"
    return factor.astype("float32")


def main() -> None:
    SymbolicTransformer = _safe_import_gplearn()

    field_rules = load_field_rules(META_PATH)
    panel = load_panel(DATA_PATH)
    cache = build_transform_cache(panel, field_rules, build_log_cache=True)
    usable_dates = cache.label.index[:-20]
    windows = get_rolling_windows(
        usable_dates,
        train_start_date=usable_dates[-(400 + 20 + 120)],
        test_start_date=usable_dates[-120],
        stride=120,
        horizon=20,
    )
    train_dates, valid_dates = windows[0]

    # Keep the baseline small. gplearn is CPU-only and operates on flattened
    # samples, so this is only a sanity baseline for the older tree-GP style.
    train_dates = train_dates[-180:]

    x_train, y_train = _build_feature_frame(cache, train_dates)
    x_valid, _ = _build_feature_frame(cache, valid_dates)

    transformer = SymbolicTransformer(
        population_size=120,
        generations=3,
        tournament_size=20,
        hall_of_fame=30,
        n_components=5,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv"),
        metric="pearson",
        parsimony_coefficient=0.001,
        max_samples=0.80,
        feature_names=list(x_train.columns),
        random_state=20260428,
        n_jobs=1,
        verbose=1,
    )
    _patch_gplearn_for_new_sklearn(transformer)

    transformer.fit(x_train.to_numpy(dtype=np.float64), y_train.to_numpy(dtype=np.float64))
    transformed_valid = transformer.transform(x_valid.to_numpy(dtype=np.float64))

    rows = []
    for i, program in enumerate(transformer._best_programs):
        factor = _program_to_factor(transformed_valid[:, i], x_valid.index)
        score = evaluate_factor(
            factor=factor,
            label=cache.label,
            tradeable=cache.tradeable,
            dates=valid_dates,
        )
        rows.append(
            {
                "component": i,
                "program": str(program),
                **{f"valid_{key}": value for key, value in score.to_dict().items()},
            }
        )

    result = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "gplearn_baseline.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("train_samples", x_train.shape)
    print("valid_samples", x_valid.shape)
    print("saved", output_path)
    print(result[["component", "valid_abs_rank_ic", "valid_ic_win_rate", "valid_ndcg_at_k", "program"]].to_string(index=False))


if __name__ == "__main__":
    main()
