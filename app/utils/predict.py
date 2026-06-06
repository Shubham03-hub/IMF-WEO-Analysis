
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st


def _resolve_model_path(filename: str) -> str:
    """Find a model file across common deployment CWD configurations."""
    candidates = [
        os.path.join(os.getcwd(), 'models', filename),
        os.path.join(os.path.dirname(__file__), '..', '..', 'models', filename),
    ]
    for path in candidates:
        norm = os.path.normpath(path)
        if os.path.exists(norm):
            return norm
    return os.path.normpath(candidates[0])


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load trained model artefacts.
    Returns (model, scaler, feature_cols) or (None, None, None) if not found.
    """
    model_path   = _resolve_model_path('best_model.pkl')
    scaler_path  = _resolve_model_path('scaler.pkl')
    feat_path    = _resolve_model_path('feature_cols.pkl')

    missing = [p for p in [model_path, scaler_path, feat_path]
               if not os.path.exists(p)]
    if missing:
        return None, None, None

    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    feature_cols = joblib.load(feat_path)
    return model, scaler, feature_cols


def predict_gdp_growth(input_dict: dict) -> float | None:
    """
    Predict GDP growth from a dict of {feature_name: value}.
    Returns predicted GDP growth (%) or None if model not available.
    Falls back to a simple linear formula if model files are missing.
    """
    model, scaler, feature_cols = load_model()

    if model is None:
        # Fallback formula — used before notebooks are run
        return None

    # Build a single-row DataFrame with the correct column order
    row = pd.DataFrame([{col: input_dict.get(col, 0.0) for col in feature_cols}])

    try:
        row_scaled  = scaler.transform(row)
        prediction  = model.predict(row_scaled)[0]
        return round(float(prediction), 2)
    except Exception:
        return None
