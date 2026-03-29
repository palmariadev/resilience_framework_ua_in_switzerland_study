# Resilience & Integration Survey – Statistical Analysis

Statistical analysis and interactive visualisation of a survey measuring resilience dimensions and socio-cultural integration factors among secondary-school students (*Matura* cohort).

---

## Analysis scope

| Notebook | Analysis |
|---|---|
| `01_violin_gender.ipynb` | Violin plots + one-way ANOVA by gender |
| `02_violin_age_group.ipynb` | Violin plots + one-way ANOVA by age group |
| `03_violin_age_gender.ipynb` | Split violin plots by age group × gender |
| `04_correlation_matrix.ipynb` | Phi-K correlation matrix across resilience and integration variables |
| `05_regression.ipynb` | Multiple regression: integration predictors → resilience outcomes |
| `06_psychometrics.ipynb` | Cronbach's Alpha reliability + inter-scale correlation analysis |

All outputs are exported as self-contained HTML files to `outputs/html/`.

---

## Data sources

Input URLs are centralised in **`src/config.py`**. Both links point to published Google Sheets CSV exports and require no authentication:

| Variable | Sheet content |
|---|---|
| `SURVEY_DATA_URL` | Aggregated dataset – one row per respondent, computed dimension scores |
| `RAW_ITEMS_URL` | Long-format raw item data – one row per respondent × survey item |

To use a different dataset, update the URLs in `src/config.py` only.

---

## Project structure

```text
.
├── src/
│   ├── config.py               # Google Sheets URL configuration
│   ├── plot_config.py          # Shared plot settings and output paths
│   ├── data_loader.py          # Survey data loading
│   ├── violin_analysis.py      # Violin plot helpers and ANOVA
│   ├── correlation_analysis.py # Correlation matrix helpers
│   ├── regression_analysis.py  # Regression helpers
│   └── psychometrics_analysis.py # Cronbach's Alpha and inter-scale correlations
├── notebooks/
│   ├── 01_violin_gender.ipynb
│   ├── 02_violin_age_group.ipynb
│   ├── 03_violin_age_gender.ipynb
│   ├── 04_correlation_matrix.ipynb
│   ├── 05_regression.ipynb
│   └── 06_psychometrics.ipynb
├── outputs/
│   └── html/                   # Exported interactive figures
├── requirements.txt
└── smart_install.py
```

---

## Setup

**Step 1 – Create a virtual environment (Python 3.12):**

```powershell
py -3.12 -m venv env
```

**Step 2 – Activate it:**

```powershell
.\env\Scripts\activate
```

**Step 3 – Install dependencies:**

```powershell
python smart_install.py
```

---

## Key libraries

| Library | Role in this project |
|---|---|
| **pandas** | Data loading, cleaning, reshaping, and grouped aggregations |
| **NumPy** | Numerical operations underlying statistical calculations |
| **Plotly** | All interactive visualisations (violin plots, heatmaps, bar charts, scatter plots); figures exported as self-contained HTML |
| **SciPy** | One-way ANOVA (`f_oneway`), Pearson correlation with p-values, F-distribution CI for Cronbach's Alpha |
| **scikit-learn** | Multiple linear regression (`LinearRegression`), model evaluation metrics |
| **statsmodels** | OLS regression with full statistical summaries (coefficients, t-tests, R²) |

---

## How this project was built

This project was developed entirely through **AI agent collaboration**. All scripting, statistical methodology, and visualisation design were produced by iterative dialogue with AI agents rather than manual coding.

The process relied on detailed **prompt harness engineering**: before writing any code, dedicated research cycles were run to establish the right approach for each component. These included:

- separate research sessions on Python library APIs and best practices (Plotly figure construction, SciPy statistical tests, pandas data pipelines)
- research on algorithm selection and statistical validity (choice of ANOVA vs. other tests, Cronbach's Alpha confidence interval method, regression diagnostics)
- research on visualisation design principles (violin plot layout, correlation heatmap colour scales, regression coefficient bar charts)

The resulting prompts were structured as precise technical specifications, which guided the agents to produce production-quality, modular, and reproducible code from the first iteration.
