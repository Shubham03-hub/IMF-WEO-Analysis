# Imports ───────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import joblib
from functools import reduce
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
CLEAN_PATH   = os.path.join(ROOT_DIR, 'data', 'processed', 'weo_clean.csv')
FEAT_PATH    = os.path.join(ROOT_DIR, 'data', 'processed', 'weo_features.csv')
MODELS_DIR   = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("Ready.")

# Load clean data ───────────────────────────────────────────────────
if not os.path.exists(CLEAN_PATH):
    raise FileNotFoundError(
        f"[ERROR] Run notebook 01 first — file not found:\n  {CLEAN_PATH}"
    )
df = pd.read_csv(CLEAN_PATH)
print(f"Shape: {df.shape}")

# Fuzzy indicator finder ───────────────────────────────────────────
def find_indicator(df, keywords):
    """Return first INDICATOR matching ALL keywords (case-insensitive)."""
    for ind in df['INDICATOR'].dropna().unique():
        if all(kw.lower() in ind.lower() for kw in keywords):
            return ind
    return None

KEY_INDICATORS_SPEC = [
    (['gdp', 'current prices', 'dollar'],              'GDP_USD'),
    (['gdp', 'constant prices', 'percent change'],     'GDP_growth_pct'),
    (['consumer price index', 'period average'],       'CPI'),
    (['unemployment rate', 'percent'],                 'Unemployment'),
    (['gross capital formation', 'percent of gdp'],    'Investment_pct'),
    (['revenue', 'general government', 'percent of gdp'], 'Gov_Revenue_pct'),
    (['expenditure', 'general government', 'percent of gdp'], 'Gov_Expenditure_pct'),
    (['population'],                                   'Population'),
]

KEY_INDICATORS = {}
for keywords, col_name in KEY_INDICATORS_SPEC:
    found = find_indicator(df, keywords)
    if found:
        KEY_INDICATORS[found] = col_name
        print(f"  ✓ {col_name}: {found}")
    else:
        print(f"  ✗ {col_name}: NOT FOUND — will be skipped")

if 'GDP_growth_pct' not in KEY_INDICATORS.values():
    raise ValueError(
        "[ERROR] GDP growth indicator not found in dataset. "
        "Check your WEO CSV — the INDICATOR column names may differ from expectations."
    )

# Pivot — one column per key indicator ──────────────────────────────
pivot_frames = []
for indicator, col_name in KEY_INDICATORS.items():
    sub = df[df['INDICATOR'] == indicator][['COUNTRY', 'YEAR', 'VALUE']].copy()
    sub.rename(columns={'VALUE': col_name}, inplace=True)
    pivot_frames.append(sub)

df_feat = reduce(lambda a, b: a.merge(b, on=['COUNTRY', 'YEAR'], how='outer'),
                 pivot_frames)
print(f"\nPivoted shape: {df_feat.shape}")
print(df_feat.head().to_string())

#  Filter to actual years (1990–2024) ────────────────────────────────
df_feat = df_feat[df_feat['YEAR'].between(1990, 2024)].copy()

# Drop rows missing the target variable
df_feat.dropna(subset=['GDP_growth_pct'], inplace=True)
print(f"After filtering years & dropping null targets: {df_feat.shape}")

# ── Cell 6: Create lag features ───────────────────────────────────────────────
df_feat.sort_values(['COUNTRY', 'YEAR'], inplace=True)

lag_cols = [c for c in ['GDP_growth_pct', 'CPI', 'Unemployment'] if c in df_feat.columns]
for col in lag_cols:
    df_feat[f'{col}_lag1'] = df_feat.groupby('COUNTRY')[col].shift(1)
    df_feat[f'{col}_lag2'] = df_feat.groupby('COUNTRY')[col].shift(2)

print(f"Lag features created for: {lag_cols}")
print(f"Shape after lag creation: {df_feat.shape}")

# Rolling features ──────────────────────────────────────────────────
if 'GDP_growth_pct' in df_feat.columns:
    df_feat['GDP_growth_3yr_avg'] = (
        df_feat.groupby('COUNTRY')['GDP_growth_pct']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    print("Rolling 3-year GDP growth average created.")

#  Encode COUNTRY ────────────────────────────────────────────────────
le = LabelEncoder()
df_feat['COUNTRY_ENC'] = le.fit_transform(df_feat['COUNTRY'].astype(str))

# Saving encoder mapping for use in Streamlit predict page
country_mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
pd.DataFrame(list(country_mapping.items()),
             columns=['COUNTRY', 'CODE']).to_csv(
    os.path.join(MODELS_DIR, 'country_encoder.csv'), index=False)
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
print(f"Encoded {df_feat['COUNTRY'].nunique()} countries.")

# Drop rows with NaN (after lag creation) ──────────────────────────
before = len(df_feat)
df_feat.dropna(inplace=True)
print(f"Dropped {before - len(df_feat):,} rows with NaN after lag creation.")
print(f"Shape after dropna: {df_feat.shape}")

if len(df_feat) < 50:
    raise ValueError(
        "[ERROR] Too few rows remain after dropping NaN. "
        "Check that your dataset has sufficient country-year coverage."
    )

# ──  Scale numeric features ──────────────────────────────────────────
EXCLUDE = ['COUNTRY', 'YEAR', 'GDP_growth_pct']
feature_cols = [c for c in df_feat.columns if c not in EXCLUDE]

scaler = MinMaxScaler()
df_feat_scaled = df_feat.copy()
df_feat_scaled[feature_cols] = scaler.fit_transform(df_feat[feature_cols])

joblib.dump(scaler,      os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.pkl'))
print(f"Scaler saved. Feature columns ({len(feature_cols)}): {feature_cols}")

#  Save feature dataset ────────────────────────────────────────────
df_feat_scaled.to_csv(FEAT_PATH, index=False)
print(f"\n✅ Saved: {FEAT_PATH}")
print(f"Final shape: {df_feat_scaled.shape}")
print(df_feat_scaled.describe().to_string())
