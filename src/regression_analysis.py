"""Regression helpers for the curated notebooks."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from src.plot_config import OUTPUT_HTML_DIR, RESILIENCE_LEVEL_COLUMNS

logger = logging.getLogger(__name__)

NUMERIC_PREDICTORS = [
    "AgeQuant",
    "EngLevelQuant",
    "CantonLangLevelQuant",
    "EducationLevelQuant",
]


def prepare_regression_dataset(survey_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare regression-ready predictors.

    Args:
        survey_df: Raw survey dataframe.

    Returns:
        Dataframe with encoded regression predictors.
    """
    logger.info("Starting regression dataset preparation.")
    regression_df = survey_df.copy()

    for column_name in ["Gender", "ArrivalStatus", "StudyStatus"]:
        if column_name in regression_df.columns:
            encoder = LabelEncoder()
            regression_df[f"{column_name}_encoded"] = encoder.fit_transform(
                regression_df[column_name].astype(str)
            )

    duration_order = [
        "Less than 1 year",
        "1-2 years",
        "2-3 years",
        "3+ years",
    ]
    return_status_order = ["No", "I don't know", "Yes"]

    if "Duration" in regression_df.columns:
        present_duration_values = [
            value
            for value in duration_order
            if value in regression_df["Duration"].dropna().unique()
        ]
        fallback_duration_values = sorted(
            value
            for value in regression_df["Duration"].dropna().unique()
            if value not in present_duration_values
        )
        duration_mapping = {
            value: index
            for index, value in enumerate(
                present_duration_values + fallback_duration_values,
                start=1,
            )
        }
        regression_df["Duration_encoded"] = regression_df["Duration"].map(duration_mapping)

    if "ReturnStatus" in regression_df.columns:
        present_return_values = [
            value
            for value in return_status_order
            if value in regression_df["ReturnStatus"].dropna().unique()
        ]
        fallback_return_values = sorted(
            value
            for value in regression_df["ReturnStatus"].dropna().unique()
            if value not in present_return_values
        )
        return_mapping = {
            value: index
            for index, value in enumerate(
                present_return_values + fallback_return_values,
                start=1,
            )
        }
        regression_df["ReturnStatus_encoded"] = regression_df["ReturnStatus"].map(return_mapping)

    for column_name in ["WorkStatus", "WorkStatusShort"]:
        if column_name in regression_df.columns:
            regression_df = pd.concat(
                [
                    regression_df,
                    pd.get_dummies(
                        regression_df[column_name],
                        prefix=column_name,
                        dtype=int,
                    ),
                ],
                axis=1,
            )

    logger.info("Completed regression dataset preparation.")
    return regression_df


def get_regression_predictors(regression_df: pd.DataFrame) -> list[str]:
    """Collect curated predictor columns for the active regression notebook.

    Args:
        regression_df: Prepared regression dataframe.

    Returns:
        Ordered list of predictor column names present in the dataframe.
    """
    candidate_columns = NUMERIC_PREDICTORS + [
        column_name
        for column_name in regression_df.columns
        if column_name.endswith("_encoded")
        or column_name.startswith("WorkStatus_")
        or column_name.startswith("WorkStatusShort_")
    ]
    return [column_name for column_name in candidate_columns if column_name in regression_df.columns]


def fit_resilience_models(
    survey_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Fit linear models for each resilience outcome.

    Args:
        survey_df: Raw survey dataframe.

    Returns:
        Dictionary of per-outcome model details and a summary dataframe.
    """
    logger.info("Starting resilience regression analysis.")
    regression_df = prepare_regression_dataset(survey_df)
    predictor_columns = get_regression_predictors(regression_df)

    model_results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for outcome_column in RESILIENCE_LEVEL_COLUMNS:
        if outcome_column not in regression_df.columns:
            continue

        available_columns = predictor_columns + [outcome_column]
        model_df = regression_df[available_columns].dropna().copy()
        if len(model_df) < 10:
            logger.warning(
                "Skipping %s because only %s complete rows are available.",
                outcome_column,
                len(model_df),
            )
            continue

        predictor_df = model_df[predictor_columns]
        outcome_series = model_df[outcome_column]
        model = LinearRegression()
        model.fit(predictor_df, outcome_series)
        r_squared = model.score(predictor_df, outcome_series)

        coefficients_df = (
            pd.DataFrame(
                {
                    "Predictor": predictor_columns,
                    "Coefficient": model.coef_,
                    "Absolute Coefficient": np.abs(model.coef_),
                }
            )
            .sort_values("Absolute Coefficient", ascending=False)
            .reset_index(drop=True)
        )

        model_results[outcome_column] = {
            "model": model,
            "r_squared": float(r_squared),
            "n_samples": int(len(model_df)),
            "coefficients": coefficients_df,
        }
        summary_rows.append(
            {
                "Outcome": outcome_column,
                "R Squared": round(float(r_squared), 3),
                "Samples": int(len(model_df)),
                "Top Predictor": coefficients_df.iloc[0]["Predictor"],
                "Top Coefficient": round(float(coefficients_df.iloc[0]["Coefficient"]), 3),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    logger.info("Completed resilience regression analysis.")
    return model_results, summary_df


def create_model_performance_figure(summary_df: pd.DataFrame) -> go.Figure:
    """Create an R-squared comparison chart for the fitted models.

    Args:
        summary_df: Regression summary dataframe.

    Returns:
        Plotly bar chart figure.
    """
    logger.info("Creating regression model performance figure.")
    if summary_df.empty:
        raise ValueError("Regression summary is empty. No performance chart can be created.")

    figure = go.Figure(
        data=[
            go.Bar(
                x=summary_df["Outcome"],
                y=summary_df["R Squared"],
                text=summary_df["R Squared"],
                textposition="auto",
            )
        ]
    )
    figure.update_layout(
        title="Regression Performance Across Resilience Outcomes",
        xaxis_title="Outcome",
        yaxis_title="R Squared",
        width=1000,
        height=600,
    )
    return figure


def create_feature_importance_heatmap(
    model_results: dict[str, dict[str, Any]],
) -> go.Figure:
    """Create a heatmap of absolute coefficients across resilience outcomes.

    Args:
        model_results: Dictionary returned by `fit_resilience_models`.

    Returns:
        Plotly heatmap figure.
    """
    logger.info("Creating regression feature-importance heatmap.")
    if not model_results:
        raise ValueError("Regression results are empty. No heatmap can be created.")

    coefficient_frames = []
    for outcome_column, result_dict in model_results.items():
        coefficient_frame = result_dict["coefficients"][
            ["Predictor", "Absolute Coefficient"]
        ].rename(columns={"Absolute Coefficient": outcome_column})
        coefficient_frames.append(coefficient_frame.set_index("Predictor"))

    heatmap_df = pd.concat(coefficient_frames, axis=1).fillna(0)
    figure = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.T.values,
            x=heatmap_df.index,
            y=heatmap_df.columns,
            colorscale="Viridis",
            text=np.round(heatmap_df.T.values, 3),
            texttemplate="%{text}",
        )
    )
    figure.update_layout(
        title="Regression Feature Importance by Outcome",
        xaxis_title="Predictor",
        yaxis_title="Outcome",
        width=1200,
        height=700,
    )
    return figure


def export_regression_figures(
    performance_figure: go.Figure,
    heatmap_figure: go.Figure,
) -> None:
    """Export the curated regression figures to HTML.

    Args:
        performance_figure: R-squared comparison bar chart.
        heatmap_figure: Feature-importance heatmap.
    """
    OUTPUT_HTML_DIR.mkdir(parents=True, exist_ok=True)
    performance_path = OUTPUT_HTML_DIR / "regression_model_performance.html"
    heatmap_path = OUTPUT_HTML_DIR / "regression_feature_importance.html"
    performance_figure.write_html(performance_path, include_plotlyjs="cdn")
    heatmap_figure.write_html(heatmap_path, include_plotlyjs="cdn")
    logger.info("Exported regression figures to %s and %s.", performance_path, heatmap_path)
