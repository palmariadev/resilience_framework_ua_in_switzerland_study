"""Central configuration for all external data source URLs.

All Google Sheets published-CSV links used by this project are defined here.
To point the analysis at a different sheet, update the URLs below.
"""

# ---------------------------------------------------------------------------
# Google Sheets – published CSV endpoints
# ---------------------------------------------------------------------------

# Aggregated survey dataset (one row per respondent, computed dimension scores)
SURVEY_DATA_URL: str = (
    "https://docs.google.com/spreadsheets/d/e/"
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/"
    "pub?gid=XXXXXXXXXXXXXXXXXXX&single=true&output=csv"
)

# Long-format raw item data (one row per respondent × survey item)
RAW_ITEMS_URL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX/"
    "export?format=csv&gid=XXXXXXXXXXXXXXXXXXXXXXXXX"
)
