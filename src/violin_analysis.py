"""Violin-plot helpers for the curated notebooks."""

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from scipy.stats import f_oneway

from src.plot_config import OUTPUT_HTML_DIR, slugify

logger = logging.getLogger(__name__)


def _describe_groups(
    plot_df: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """Create grouped summary statistics for violin notebooks.

    Args:
        plot_df: Filtered dataframe used for plotting.
        value_column: Numeric variable being plotted.
        group_columns: Columns used to group rows.

    Returns:
        Summary statistics dataframe.
    """
    summary_df = (
        plot_df.groupby(group_columns)[value_column]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "Count",
                "mean": "Mean",
                "median": "Median",
                "std": "Std Dev",
                "min": "Min",
                "max": "Max",
            }
        )
    )
    return summary_df.round(3)


def perform_group_anova(
    survey_df: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> dict[str, Any]:
    """Run one-way ANOVA for one outcome across one grouping variable.

    Args:
        survey_df: Input survey dataframe.
        value_column: Numeric outcome column.
        group_column: Grouping column.

    Returns:
        Dictionary with ANOVA results and grouped summary statistics.
    """
    logger.info(
        "Running ANOVA for %s grouped by %s.",
        value_column,
        group_column,
    )
    plot_df = survey_df[[group_column, value_column]].dropna().copy()
    grouped_values = [
        group_df[value_column].to_numpy()
        for _, group_df in plot_df.groupby(group_column)
        if not group_df.empty
    ]

    if len(grouped_values) < 2:
        raise ValueError(f"Need at least two groups for {group_column}.")

    f_statistic, p_value = f_oneway(*grouped_values)
    summary_df = _describe_groups(plot_df, value_column, [group_column])

    logger.info(
        "Completed ANOVA for %s grouped by %s.",
        value_column,
        group_column,
    )
    return {
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "summary_table": summary_df,
        "n_rows": int(len(plot_df)),
    }


def create_gender_violin_figure(
    survey_df: pd.DataFrame,
    value_column: str,
) -> tuple[go.Figure, pd.DataFrame, dict[str, Any]]:
    """Create violin plot by gender.

    Args:
        survey_df: Input survey dataframe.
        value_column: Numeric outcome column.

    Returns:
        Plotly figure, grouped summary table, and ANOVA results.
    """
    anova_results = perform_group_anova(survey_df, value_column, "Gender")
    plot_df = survey_df[["Gender", value_column]].dropna().copy()

    figure = go.Figure()
    colors = {
        "Female": "rgba(214, 39, 40, 0.55)",
        "Male": "rgba(31, 119, 180, 0.55)",
    }

    for gender_value in sorted(plot_df["Gender"].astype(str).unique()):
        gender_df = plot_df[plot_df["Gender"] == gender_value]
        figure.add_trace(
            go.Violin(
                x=gender_df["Gender"],
                y=gender_df[value_column],
                name=gender_value,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.2,
                marker=dict(size=5, opacity=0.65),
                line_color=colors.get(gender_value, "rgba(90, 90, 90, 1)"),
                fillcolor=colors.get(gender_value, "rgba(90, 90, 90, 0.45)"),
            )
        )

    figure.update_layout(
        title=f"{value_column} by Gender",
        xaxis_title="Gender",
        yaxis_title=value_column,
        violinmode="group",
        width=900,
        height=600,
    )
    return figure, anova_results["summary_table"], anova_results


def create_age_group_violin_figure(
    survey_df: pd.DataFrame,
    value_column: str,
) -> tuple[go.Figure, pd.DataFrame, dict[str, Any]]:
    """Create violin plot by age group.

    Args:
        survey_df: Input survey dataframe.
        value_column: Numeric outcome column.

    Returns:
        Plotly figure, grouped summary table, and ANOVA results.
    """
    anova_results = perform_group_anova(survey_df, value_column, "Age")
    plot_df = survey_df[["Age", value_column]].dropna().copy()

    figure = go.Figure()
    for age_group, age_df in plot_df.groupby("Age", sort=True):
        figure.add_trace(
            go.Violin(
                x=age_df["Age"],
                y=age_df[value_column],
                name=str(age_group),
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.2,
                marker=dict(size=4, opacity=0.6),
            )
        )

    figure.update_layout(
        title=f"{value_column} by Age Group",
        xaxis_title="Age Group",
        yaxis_title=value_column,
        violinmode="group",
        width=1000,
        height=650,
        showlegend=False,
    )
    return figure, anova_results["summary_table"], anova_results


def create_age_gender_violin_figure(
    survey_df: pd.DataFrame,
    value_column: str,
) -> tuple[go.Figure, pd.DataFrame]:
    """Create split violin plot by age group and gender.

    Args:
        survey_df: Input survey dataframe.
        value_column: Numeric outcome column.

    Returns:
        Plotly figure and grouped summary table.
    """
    logger.info("Creating age and gender violin plot for %s.", value_column)
    plot_df = survey_df[["Age", "Gender", value_column]].dropna().copy()
    summary_df = _describe_groups(plot_df, value_column, ["Age", "Gender"])

    colors = {
        "Female": ("rgba(214, 39, 40, 0.5)", "negative"),
        "Male": ("rgba(31, 119, 180, 0.5)", "positive"),
    }
    figure = go.Figure()

    for gender_value in ["Female", "Male"]:
        gender_df = plot_df[plot_df["Gender"] == gender_value]
        if gender_df.empty:
            continue

        fill_color, side_value = colors.get(
            gender_value,
            ("rgba(90, 90, 90, 0.5)", "positive"),
        )
        figure.add_trace(
            go.Violin(
                x=gender_df["Age"],
                y=gender_df[value_column],
                name=gender_value,
                legendgroup=gender_value,
                scalegroup=gender_value,
                side=side_value,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.15,
                marker=dict(size=4, opacity=0.65),
                line_color=fill_color.replace("0.5", "1"),
                fillcolor=fill_color,
            )
        )

    figure.update_layout(
        title=f"{value_column} by Age Group and Gender",
        xaxis_title="Age Group",
        yaxis_title=value_column,
        violinmode="overlay",
        width=1000,
        height=650,
    )
    return figure, summary_df


def export_figure_html(figure: go.Figure, filename: str) -> str:
    """Export a Plotly figure to the standard HTML output folder.

    Args:
        figure: Plotly figure to export.
        filename: Output filename.

    Returns:
        Exported path as a string.
    """
    output_path = OUTPUT_HTML_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")
    logger.info("Exported HTML figure to %s.", output_path)
    return str(output_path)


def build_violin_filename(prefix: str, value_column: str) -> str:
    """Build a standard HTML filename for a violin plot.

    Args:
        prefix: Plot family prefix.
        value_column: Numeric outcome column.

    Returns:
        Standardized filename.
    """
    return f"{prefix}_{slugify(value_column)}.html"
