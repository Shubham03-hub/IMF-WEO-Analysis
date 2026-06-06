───────────────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from utils.predict import load_model, predict_gdp_growth

st.set_page_config(page_title="Predict GDP Growth", page_icon="🤖", layout="wide")
st.title("🤖 GDP Growth Predictor")
st.markdown("Forecast a country's GDP growth rate using macroeconomic indicators.")

# ── Check model availability ──────────────────────────────────────────────────
model, scaler, feature_cols = load_model()
model_available = model is not None

if not model_available:
    st.warning(
        "⚠️ **Trained model not found.**\n\n"
        "The prediction below uses a simplified formula until you run "
        "Notebooks 03 and 04 to train and save the model (`models/best_model.pkl`).\n\n"
        "Run notebooks in order: `01 → 02 → 03 → 04` then refresh this page."
    )
else:
    st.success("✅ Trained XGBoost model loaded successfully.")

st.markdown("---")

# ── Input sliders ─────────────────────────────────────────────────────────────
st.subheader("Input Economic Indicators")
col1, col2 = st.columns(2)

with col1:
    cpi          = st.slider("CPI — Inflation Index",              50.0,  500.0, 100.0, step=1.0)
    unemployment = st.slider("Unemployment Rate (%)",               0.0,   40.0,   5.0, step=0.1)
    investment   = st.slider("Gross Capital Formation (% of GDP)", 0.0,   60.0,  22.0, step=0.5)

with col2:
    gov_revenue  = st.slider("Government Revenue (% of GDP)",      5.0,   60.0,  30.0, step=0.5)
    gov_expense  = st.slider("Government Expenditure (% of GDP)",  5.0,   70.0,  35.0, step=0.5)
    population   = st.slider("Population (millions)",               0.1, 1500.0,  50.0, step=1.0)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict GDP Growth", type="primary", use_container_width=True):

    if model_available:
        # Build input dict using the exact feature names the model was trained on
        input_values = {
            'CPI':                 cpi,
            'Unemployment':        unemployment,
            'Investment_pct':      investment,
            'Gov_Revenue_pct':     gov_revenue,
            'Gov_Expenditure_pct': gov_expense,
            'Population':          population,
            # Lag features: pre-fill with current values as approximation
            'GDP_growth_pct_lag1': 2.5,
            'GDP_growth_pct_lag2': 2.5,
            'CPI_lag1':            cpi,
            'CPI_lag2':            cpi,
            'Unemployment_lag1':   unemployment,
            'Unemployment_lag2':   unemployment,
            'GDP_growth_3yr_avg':  2.5,
        }
        # Fill any remaining features with 0
        for fc in feature_cols:
            if fc not in input_values:
                input_values[fc] = 0.0

        growth_estimate = predict_gdp_growth(input_values)
        source_label    = "XGBoost Model"
    else:
        # Fallback formula
        growth_estimate = (
            0.03 * investment
            - 0.15 * unemployment
            - 0.001 * cpi
            + 0.05 * gov_revenue
            - 0.03 * gov_expense
            + 2.0
        )
        source_label = "Simplified Formula (model not yet trained)"

    if growth_estimate is not None:
        st.markdown("---")
        st.subheader("Prediction Result")

        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if growth_estimate > 3:
                st.success(f"### 📈 Predicted GDP Growth: **{growth_estimate:.2f}%**")
                st.info("**Strong growth expected.** Indicators are broadly positive.")
            elif growth_estimate > 1:
                st.warning(f"### 📊 Predicted GDP Growth: **{growth_estimate:.2f}%**")
                st.info("**Moderate growth expected.** Economy expanding but gradually.")
            elif growth_estimate > 0:
                st.warning(f"### 📉 Predicted GDP Growth: **{growth_estimate:.2f}%**")
                st.info("**Slow growth.** Economy is expanding but fragile.")
            else:
                st.error(f"### ⚠️ Predicted GDP Growth: **{growth_estimate:.2f}%**")
                st.info("**Contraction risk detected.** Negative growth forecast.")

            st.caption(f"Source: {source_label}")

        with gauge_col:
            # ── Simple indicator summary ─────────────────────────────
            fiscal_balance = gov_revenue - gov_expense
            st.metric("Fiscal Balance (% GDP)", f"{fiscal_balance:.1f}%",
                      delta_color="normal")
            st.metric("Predicted GDP Growth",   f"{growth_estimate:.2f}%",
                      delta=f"{growth_estimate - 2.5:.2f}% vs 2.5% baseline")
    else:
        st.error("Prediction failed. Check model files and try again.")
