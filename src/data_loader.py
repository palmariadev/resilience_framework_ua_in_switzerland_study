"""Load the published survey dataset for the active notebooks."""

import logging

import pandas as pd

from src.plot_config import DEFAULT_DATA_URL

logger = logging.getLogger(__name__)


def load_survey_data(url: str = DEFAULT_DATA_URL) -> pd.DataFrame:
    """Load survey data from the published Google Sheets CSV.

    Args:
        url: Published CSV URL for the survey data.

    Returns:
        Loaded survey dataframe.

    Raises:
        Exception: Propagates the underlying read error.
    """
    logger.info("Starting survey data load.")
    try:
        survey_df = pd.read_csv(url)
        logger.info(
            "Survey data loaded successfully with %s rows and %s columns.",
            survey_df.shape[0],
            survey_df.shape[1],
        )
        return survey_df
    except Exception:
        logger.exception("Survey data load failed.")
        raise
