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
    "2PACX-1vRSEgWEyMJUR-PtAP2NZ2kvJJWG3HBotjDFvWF_bG4zZhKdasgodmgt9i3UFIDuO5I3kKNX2tO1wGcv/"
    "pub?gid=458419167&single=true&output=csv"
)

# Long-format raw item data (one row per respondent × survey item)
RAW_ITEMS_URL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1l6a5iu4p4YxNdZ6ECdB56BJAWOWQoUDtBVjgmCBfOIA/"
    "export?format=csv&gid=2132539378"
)
