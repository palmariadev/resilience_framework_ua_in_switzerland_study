"""Correlation analysis helpers for resilience and integration variables."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.plot_config import OUTPUT_HTML_DIR

logger = logging.getLogger(__name__)

# Maps internal column names to clean publication labels.
# The insertion order defines the variable order in the published matrix.
DISPLAY_LABELS: dict[str, str] = {
    "EngLevelQuant": "English Level",
    "CantonLangLevelQuant": "Canton Language Level",
    "EducationLevelQuant": "Education Level",
    "Adversities Level": "Adversities",
    "Bounce Back Level": "Bounce Back",
    "Bounce Forward Level": "Bounce Forward",
    "Environment Support Level": "Environment Support",
    "Mindset Level": "Mindset",
    "AgeQuant": "Age",
    "Gender_encoded": "Gender",
    "Duration_encoded": "Duration in CH",
    "ArrivalStatus_encoded": "Arrived Alone",
    "StudyStatus_encoded": "Studying",
    "ReturnStatus_encoded": "Return Intention",
}

# Column selection order is derived from DISPLAY_LABELS — single source of truth.
CORRELATION_COLUMNS: list[str] = list(DISPLAY_LABELS)

# Binary variables encoded alphabetically as 0/1 (no external dependency needed).
_BINARY_COLUMNS: tuple[str, ...] = ("Gender", "ArrivalStatus", "StudyStatus")

# Ordinal mappings in chronological / logical order.
_DURATION_ORDER: dict[str, int] = {
    "Less than 6 months": 1,
    "From 6 months to 1 year": 2,
    "1 - 2 years": 3,
    "2 - 3 years": 4,
    "3 - 3.5 years": 5,
}

_RETURN_STATUS_ORDER: dict[str, int] = {
    "No": 1,
    "I don't know": 2,
    "Yes": 3,
}


def _encode_categorical_variables(survey_df: pd.DataFrame) -> pd.DataFrame:
    """Add encoded columns for categorical variables.

    Binary variables are encoded alphabetically as 0/1 using pandas
    Categorical codes. Ordinal variables use the domain-specific orderings
    defined in the module constants.

    Args:
        survey_df: Raw survey dataframe.

    Returns:
        Copy of the dataframe with encoded columns appended.
    """
    encoded_df = survey_df.copy()

    for column_name in _BINARY_COLUMNS:
        if column_name in encoded_df.columns:
            encoded_df[f"{column_name}_encoded"] = (
                pd.Categorical(encoded_df[column_name].astype(str)).codes
            )

    if "Duration" in encoded_df.columns:
        encoded_df["Duration_encoded"] = encoded_df["Duration"].map(_DURATION_ORDER)

    if "ReturnStatus" in encoded_df.columns:
        encoded_df["ReturnStatus_encoded"] = encoded_df["ReturnStatus"].map(_RETURN_STATUS_ORDER)

    return encoded_df


def calculate_correlation_matrix(survey_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Pearson correlation matrix for resilience and integration variables.

    Args:
        survey_df: Raw survey dataframe.

    Returns:
        Pearson correlation matrix for the selected variables.
    """
    logger.info("Starting correlation matrix calculation.")
    encoded_df = _encode_categorical_variables(survey_df)
    available_columns = [col for col in CORRELATION_COLUMNS if col in encoded_df.columns]
    correlation_df = encoded_df[available_columns].corr(numeric_only=True)
    logger.info("Completed correlation matrix calculation.")
    return correlation_df


def create_correlation_heatmap(correlation_df: pd.DataFrame) -> go.Figure:
    """Create a Plotly heatmap for the curated correlation matrix.

    The y-axis is reversed so variables read top-to-bottom, matching the
    x-axis left-to-right order, with the diagonal running top-left to
    bottom-right as in the standard academic format.

    Args:
        correlation_df: Pearson correlation matrix.

    Returns:
        Plotly heatmap figure.
    """
    logger.info("Creating correlation heatmap figure.")
    display_columns = [DISPLAY_LABELS.get(col, col) for col in correlation_df.columns]
    figure = go.Figure(
        data=go.Heatmap(
            z=correlation_df.values,
            x=display_columns,
            y=display_columns,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(correlation_df.values, 2),
            texttemplate="%{text}",
            hoverongaps=False,
        )
    )
    figure.update_layout(
        title="Correlation Matrix: Resilience and Integration Variables",
        width=1000,
        height=850,
        xaxis=dict(title="Variables", tickangle=45),
        yaxis=dict(title="Variables", autorange="reversed"),
    )
    return figure


def summarize_strong_correlations(
    correlation_df: pd.DataFrame,
    limit: int = 10,
) -> pd.DataFrame:
    """Return the strongest unique pairwise correlations.

    Uses the upper triangle of the matrix to avoid duplicate pairs and
    self-correlations. Variable names are mapped to display labels.

    Args:
        correlation_df: Pearson correlation matrix.
        limit: Maximum number of pairs to return.

    Returns:
        Dataframe of strongest pairwise correlations sorted by absolute value.
    """
    logger.info("Summarizing strongest correlations.")
    mask = np.triu(np.ones(correlation_df.shape, dtype=bool), k=1)
    summary_df = (
        correlation_df.where(mask)
        .stack()
        .rename("Correlation")
        .reset_index()
        .rename(columns={"level_0": "Variable A", "level_1": "Variable B"})
        .assign(**{"Absolute Correlation": lambda df: df["Correlation"].abs()})
        .sort_values("Absolute Correlation", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    summary_df["Variable A"] = summary_df["Variable A"].map(
        lambda col: DISPLAY_LABELS.get(col, col)
    )
    summary_df["Variable B"] = summary_df["Variable B"].map(
        lambda col: DISPLAY_LABELS.get(col, col)
    )
    summary_df["Correlation"] = summary_df["Correlation"].round(3)
    summary_df["Absolute Correlation"] = summary_df["Absolute Correlation"].round(3)
    logger.info("Completed correlation summary table.")
    return summary_df


def export_correlation_heatmap(figure: go.Figure) -> None:
    """Export the curated correlation heatmap to HTML.

    Args:
        figure: Plotly heatmap figure.
    """
    output_path = OUTPUT_HTML_DIR / "correlation_matrix_resilience_integration.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")
    logger.info("Exported correlation heatmap to %s.", output_path)
