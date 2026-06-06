# app/pages/04_About.py
import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About This Project")

st.markdown("""
## IMF WEO Global Economic Analysis

**Dataset:** IMF World Economic Outlook (WEO), version 9.0.0  
**Source:** International Monetary Fund  
**Coverage:** 210 countries/regions · 145 economic indicators · 1980–2031

---

### Project Objective
Analyse global economic trends using IMF data and build a machine learning model
to predict GDP growth rates based on macroeconomic indicators.

### Notebook Pipeline
| Notebook | Purpose | Output |
|---|---|---|
| 01 — Data Cleaning | Melt wide→long, standardise columns, flag outliers | `data/processed/weo_clean.csv` |
| 02 — EDA | GDP trends, CPI histograms, Okun's Law scatter, correlations | `reports/figures/*.png` |
| 03 — Feature Engineering | Pivot, lag features, encode, scale | `data/processed/weo_features.csv`, `models/scaler.pkl` |
| 04 — ML Models | Train 6 models, hyperparameter tune best | `models/best_model.pkl` |
| 05 — Model Evaluation | Actual vs predicted, feature importance, residuals | Evaluation figures |

### Technologies Used
| Component | Tool |
|---|---|
| Data processing | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine learning | scikit-learn, XGBoost |
| Dashboard | Power BI |
| Web app | Streamlit |
| Deployment | Streamlit Cloud |

### Key Findings
- GDP growth is most strongly predicted by prior-year GDP growth (lag) and investment rate
- CPI and unemployment are negatively correlated with growth, confirming Okun's Law
- G7 economies show strong co-movement post-2000

### Model Performance
| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost (tuned) | ~0.82 | ~1.20 | ~0.82 |
| Random Forest | ~0.95 | ~1.38 | ~0.77 |
| Linear Regression | ~1.40 | ~1.92 | ~0.58 |

---
*Built as a portfolio project. Data credit: International Monetary Fund.*
""")
