
import sys
import os

# ── Ensure app/ and app/utils/ are importable regardless of CWD ──────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st

st.set_page_config(
    page_title="IMF WEO Economic Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌍 IMF WEO Analysis")
st.sidebar.markdown("Global Economic Outlook Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Navigate using the **Pages** listed above in the sidebar.\n\n"
    "- 📊 **Home** — Dataset overview\n"
    "- 🔍 **EDA** — Explore trends & correlations\n"
    "- 🤖 **Predict** — Forecast GDP growth\n"
    "- ℹ️ **About** — Project details"
)
st.sidebar.markdown("---")
st.sidebar.info("Dataset: IMF WEO 9.0.0 · 210 countries · 1980–2031")

# ── Main page ─────────────────────────────────────────────────────────────────
st.title("🌍 IMF World Economic Outlook")
st.subheader("Global Economic Analysis & GDP Growth Forecasting")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries",     "210")
col2.metric("Indicators",    "145")
col3.metric("Year Range",    "1980–2031")
col4.metric("Total Records", "8,200+")

st.markdown("---")
st.markdown(
    "### Getting Started\n"
    "Use the **sidebar** to navigate between pages:\n\n"
    "| Page | Description |\n"
    "|---|---|\n"
    "| 📊 Home | Dataset summary and quick stats |\n"
    "| 🔍 EDA | GDP trends, inflation, correlations |\n"
    "| 🤖 Predict | Forecast GDP growth with sliders |\n"
    "| ℹ️ About | Methodology and technologies |\n"
)

st.markdown("---")
st.markdown(
    "> **First time?** Make sure you've run the notebooks in order "
    "(`01 → 02 → 03 → 04 → 05`) to generate the processed data and model files."
)
