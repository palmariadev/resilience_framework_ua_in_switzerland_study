"""Psychometric analysis for the Matura resilience survey.

Provides two core analyses that justify aggregating individual survey items
into four composite dimension scores:

1. **Cronbach's Alpha** – confirms items within each scale are internally
   consistent enough to be averaged into a single score.
2. **Inter-scale correlations** – confirms the four scales are related but
   distinct, justifying four separate scores rather than one index.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RAW_ITEMS_URL as RAW_LONG_URL

logger = logging.getLogger(__name__)

RESILIENCE_SCALES: list[str] = [
    "Bounce Back",
    "Bounce Forward",
    "Environment",
    "Mindset",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_long_format_data() -> pd.DataFrame:
    """Download and lightly clean the long-format raw item data.

    Returns:
        DataFrame with one row per (respondent × item), including columns
        ``ResponseID``, ``QuestionID``, ``Resilience Dimension``, and
        ``NumericAdjustedAnswer`` (polarity-adjusted, 1–5 scale).
    """
    logger.info("Loading long-format raw item data from Google Sheets…")
    df = pd.read_csv(RAW_LONG_URL, encoding="utf-8")
    logger.info("Loaded: %d rows, %d columns", *df.shape)

    df["Resilience Dimension"] = df["Resilience Dimension"].str.strip()
    df["QuestionID"] = pd.to_numeric(df["QuestionID"], errors="coerce")
    df["NumericAdjustedAnswer"] = pd.to_numeric(
        df["NumericAdjustedAnswer"], errors="coerce"
    )
    return df


def build_item_matrices(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot long-format data into one wide item matrix per scale.

    Args:
        long_df: DataFrame returned by :func:`load_long_format_data`.

    Returns:
        Dictionary mapping scale name → DataFrame of shape
        (n_respondents, n_items).  Respondents with any missing item are
        dropped listwise.
    """
    matrices: dict[str, pd.DataFrame] = {}
    scale_rows = long_df[long_df["Resilience Dimension"].isin(RESILIENCE_SCALES)]

    for scale in RESILIENCE_SCALES:
        pivot = (
            scale_rows[scale_rows["Resilience Dimension"] == scale]
            .pivot_table(
                index="ResponseID",
                columns="QuestionID",
                values="NumericAdjustedAnswer",
                aggfunc="first",
            )
            .dropna()
        )
        pivot.columns = [f"Q{int(c)}" for c in pivot.columns]
        logger.info("'%s': %d respondents × %d items", scale, *pivot.shape)
        matrices[scale] = pivot

    return matrices


# ---------------------------------------------------------------------------
# 1. Cronbach's Alpha
# ---------------------------------------------------------------------------


def compute_cronbach_alpha(item_matrix: pd.DataFrame) -> dict[str, Any]:
    """Compute Cronbach's Alpha and its 95 % confidence interval.

    Uses the standard formula α = (k/(k−1)) × (1 − Σvar_i / var_total) and
    the Feldt, Woodruff & Salih (1987) asymptotic CI via the F-distribution.

    Args:
        item_matrix: DataFrame of shape (n, k) with numeric item scores.

    Returns:
        Dictionary with keys ``alpha``, ``ci_lower``, ``ci_upper``,
        ``n_items``, ``n_obs``, and ``interpretation``.
    """
    matrix = item_matrix.values.astype(float)
    n, k = matrix.shape
    item_variances = matrix.var(axis=0, ddof=1)
    total_variance = matrix.sum(axis=1).var(ddof=1)

    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)

    f_lower = stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1))
    f_upper = stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1))
    ci_lower = 1 - (1 - alpha) / f_lower
    ci_upper = 1 - (1 - alpha) / f_upper

    interp = (
        "Excellent (≥ .90)" if alpha >= 0.90
        else "Good (.80–.89)"        if alpha >= 0.80
        else "Acceptable (.70–.79)"  if alpha >= 0.70
        else "Questionable (.60–.69)" if alpha >= 0.60
        else "Poor (< .60)"
    )

    return {
        "alpha":          round(float(alpha),    3),
        "ci_lower":       round(float(ci_lower), 3),
        "ci_upper":       round(float(ci_upper), 3),
        "n_items":        k,
        "n_obs":          n,
        "interpretation": interp,
    }


def build_reliability_summary(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble a reliability summary table for all scales.

    Args:
        matrices: Dictionary from :func:`build_item_matrices`.

    Returns:
        DataFrame with one row per scale and columns for k, N, α, 95 % CI,
        and interpretation.
    """
    rows: list[dict[str, Any]] = []
    for scale, mat in matrices.items():
        res = compute_cronbach_alpha(mat)
        rows.append({
            "Scale":          scale,
            "k (items)":      res["n_items"],
            "N":              res["n_obs"],
            "α":              res["alpha"],
            "95 % CI":        f"[{res['ci_lower']}, {res['ci_upper']}]",
            "Interpretation": res["interpretation"],
        })
    return pd.DataFrame(rows).set_index("Scale")


# ---------------------------------------------------------------------------
# 2. Inter-scale correlations
# ---------------------------------------------------------------------------


def compute_scale_means(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute mean composite score per respondent for each scale.

    Args:
        matrices: Dictionary from :func:`build_item_matrices`.

    Returns:
        DataFrame (respondents × scales) with mean item scores, restricted to
        respondents present in all scales.
    """
    mean_scores = {scale: mat.mean(axis=1).rename(scale) for scale, mat in matrices.items()}
    combined = pd.concat(mean_scores.values(), axis=1, join="inner")
    logger.info("Scale means: %d respondents × %d scales", *combined.shape)
    return combined


def compute_inter_scale_correlations(
    scale_means: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson inter-scale correlation matrix with p-values.

    Args:
        scale_means: DataFrame (n × scales) from :func:`compute_scale_means`.

    Returns:
        Tuple of (r_matrix, p_matrix) – DataFrames indexed and columned by
        scale name, values rounded to three decimal places.
    """
    cols = scale_means.columns.tolist()
    n = len(cols)
    r_data = np.ones((n, n))
    p_data = np.zeros((n, n))

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i != j:
                r, p = stats.pearsonr(scale_means[col_i].values, scale_means[col_j].values)
                r_data[i, j] = r
                p_data[i, j] = p

    r_df = pd.DataFrame(np.round(r_data, 3), index=cols, columns=cols)
    p_df = pd.DataFrame(np.round(p_data, 3), index=cols, columns=cols)
    return r_df, p_df
