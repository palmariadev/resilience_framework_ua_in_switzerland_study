"""Shared configuration for notebooks, plots, and exports."""

from pathlib import Path

from src.config import SURVEY_DATA_URL as DEFAULT_DATA_URL

RESILIENCE_LEVEL_COLUMNS = [
    "Adversities Level",
    "Bounce Back Level",
    "Bounce Forward Level",
    "Environment Support Level",
    "Mindset Level",
]

INTEGRATION_COLUMNS = [
    "EngLevelQuant",
    "CantonLangLevelQuant",
    "EducationLevelQuant",
    "AgeQuant",
]

OUTPUT_HTML_DIR = Path("outputs") / "html"


def slugify(value: str) -> str:
    """Convert a label to a filesystem-friendly slug.

    Args:
        value: Human-readable label.

    Returns:
        Lowercase slug safe for filenames.
    """
    return (
        value.strip()
        .lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
    )
