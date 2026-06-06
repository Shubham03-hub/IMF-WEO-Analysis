
# Data loading utilities for the Streamlit app.
# Handles both local dev paths and Streamlit Cloud deployment paths.

import os
import pandas as pd
import streamlit as st

def _resolve_path(relative_path: str) -> str:
    """
    Try multiple base directories to locate a file.
    This handles:
      - Running `streamlit run app/main.py` from the project root
      - Running `streamlit run main.py` from inside app/
      - Streamlit Cloud deployment (CWD is the repo root)
    """
    candidates = [
        # CWD-relative (most common: running from project root)
        os.path.join(os.getcwd(), relative_path),
        # One level up from this file's directory (app/ → project root)
        os.path.join(os.path.dirname(__file__), '..', '..', relative_path),
        # Absolute from project root embedded in path
        os.path.join(os.path.dirname(__file__), relative_path),
    ]
    for path in candidates:
        norm = os.path.normpath(path)
        if os.path.exists(norm):
            return norm
    # Return first candidate (will trigger FileNotFoundError with a clear message)
    return os.path.normpath(candidates[0])


@st.cache_data(show_spinner="Loading dataset…")
def load_clean_data() -> pd.DataFrame:
    path = _resolve_path('data/processed/weo_clean.csv')
    if not os.path.exists(path):
        st.error(
            f"**weo_clean.csv not found.**\n\n"
            f"Expected at: `{path}`\n\n"
            "Please run **Notebook 01** first to generate the cleaned dataset."
        )
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading feature data…")
def load_features() -> pd.DataFrame:
    path = _resolve_path('data/processed/weo_features.csv')
    if not os.path.exists(path):
        st.error(
            f"**weo_features.csv not found.**\n\n"
            f"Expected at: `{path}`\n\n"
            "Please run **Notebook 03** first to generate features."
        )
        st.stop()
    return pd.read_csv(path)


def get_countries(df: pd.DataFrame) -> list:
    return sorted(df['COUNTRY'].dropna().unique().tolist())


def get_indicators(df: pd.DataFrame) -> list:
    return sorted(df['INDICATOR'].dropna().unique().tolist())
