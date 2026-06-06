
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to THIS file so the script works from any CWD
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # notebooks/
ROOT_DIR   = os.path.dirname(BASE_DIR)                    # project root
RAW_PATH   = os.path.join(ROOT_DIR, 'data', 'raw',       'weo_dataset.csv')
CLEAN_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'weo_clean.csv')

print("Libraries loaded.")
print(f"Expecting raw data at: {RAW_PATH}")

#  Load raw data ─────────────────────────────────────────────────────
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Raw dataset not found at:\n  {RAW_PATH}\n"
        "Copy your weo_dataset.csv into data/raw/ and re-run."
    )

df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Shape: {df.shape}")
print(f"Columns (first 10): {df.columns.tolist()[:10]} ...")
print(df.head(3).to_string())

# Identify year columns ─────────────────────────────────────────────
# Year columns are purely numeric strings like '1980', '1981', …
year_cols = [c for c in df.columns if str(c).strip().isdigit()]
meta_cols  = [c for c in df.columns if c not in year_cols]
print(f"\nYear columns: {year_cols[0]} → {year_cols[-1]}  ({len(year_cols)} years)")
print(f"Meta columns: {meta_cols}")

#  Melt wide → long format ──────────────────────────────────────────
df_long = df.melt(
    id_vars=meta_cols,
    value_vars=year_cols,
    var_name='YEAR',
    value_name='VALUE'
)
df_long['YEAR']  = pd.to_numeric(df_long['YEAR'],  errors='coerce').astype('Int64')
df_long['VALUE'] = pd.to_numeric(df_long['VALUE'], errors='coerce')
print(f"\nLong format shape: {df_long.shape}")
print(df_long.head().to_string())

# Standardise column names ─────────────────────────────────────────
# The WEO CSV may use slightly different header names across releases.
# We try common variants and rename to a stable set.
rename_map = {}
for col in df_long.columns:
    col_upper = col.strip().upper()
    # Country
    if col_upper in ('COUNTRY', 'COUNTRYNAME', 'ECONOMY'):
        rename_map[col] = 'COUNTRY'
    # Indicator / Subject
    elif col_upper in ('WEO SUBJECT CODE', 'SUBJECT', 'INDICATOR_CODE',
                       'SERIES_CODE', 'WEOSUBJECTCODE'):
        rename_map[col] = 'INDICATOR_CODE'
    elif col_upper in ('SUBJECT DESCRIPTOR', 'INDICATOR', 'INDICATORNAME',
                       'SERIES_NAME', 'SUBJECTDESCRIPTOR'):
        rename_map[col] = 'INDICATOR'
    # Topic / Group
    elif col_upper in ('SUBJECT NOTES', 'TOPIC', 'GROUP', 'SUBJECTNOTES'):
        rename_map[col] = 'TOPIC'
    # Unit
    elif col_upper in ('UNITS', 'UNIT', 'UNITNAME'):
        rename_map[col] = 'UNIT'
    # Scale
    elif col_upper in ('SCALE', 'SCALENAME'):
        rename_map[col] = 'SCALE'

if rename_map:
    df_long.rename(columns=rename_map, inplace=True)
    print(f"\nRenamed columns: {rename_map}")

# Ensure required columns exist; create empty ones if missing so downstream
# cells don't crash regardless of which WEO release is used.
for required in ['COUNTRY', 'INDICATOR', 'TOPIC', 'UNIT', 'SCALE']:
    if required not in df_long.columns:
        df_long[required] = np.nan
        print(f"  [WARN] Column '{required}' not found — filled with NaN.")

# Missing values ────────────────────────────────────────────────────
missing = df_long['VALUE'].isna().sum()
total   = len(df_long)
print(f"\nMissing VALUE:  {missing:,} / {total:,}  ({missing/total*100:.1f}%)")

# Future projection years (2025–2031) legitimately have NaN — keep them.
# Drop only rows where the key identifier fields are missing.
df_long.dropna(subset=['COUNTRY', 'INDICATOR'], inplace=True)
print(f"After dropping missing COUNTRY/INDICATOR: {df_long.shape}")

#  Remove duplicates ─────────────────────────────────────────────────
before = len(df_long)
df_long.drop_duplicates(subset=['COUNTRY', 'INDICATOR', 'YEAR'],
                         keep='first', inplace=True)
after = len(df_long)
print(f"Duplicates removed: {before - after:,}")

# Fix mixed-type string columns ─────────────────────────────────────
mixed_cols = [c for c in ['INT_ACC_ITEM', 'COMMODITY', 'SOC_CONCEPTS',
                           'ACCOUNTING_ENTRY', 'BASE_YEAR',
                           'LATEST_ACTUAL_ANNUAL_DATA']
              if c in df_long.columns]
for col in mixed_cols:
    df_long[col] = df_long[col].astype(str).replace({'nan': np.nan, 'None': np.nan})
print("Mixed-type columns fixed.")

# Keep only key columns ────────────────────────────────────────────
keep = [c for c in ['COUNTRY', 'INDICATOR', 'TOPIC', 'YEAR', 'VALUE', 'UNIT', 'SCALE']
        if c in df_long.columns]
df_clean = df_long[keep].copy()
df_clean.reset_index(drop=True, inplace=True)
print(f"\nClean dataframe shape: {df_clean.shape}")
print(df_clean.sample(5).to_string())

# Flag outliers (IQR method, per indicator) ────────────────────────

def flag_outliers(group):
    q1  = group['VALUE'].quantile(0.25)
    q3  = group['VALUE'].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    group = group.copy()
    group['IS_OUTLIER'] = ~group['VALUE'].between(lower, upper)
    return group

df_clean = (df_clean
            .groupby('INDICATOR', group_keys=False)
            .apply(flag_outliers))
print(f"Outliers flagged (not removed): {df_clean['IS_OUTLIER'].sum():,}")

# Save clean data 
os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
df_clean.to_csv(CLEAN_PATH, index=False)
print(f"\n✅ Saved: {CLEAN_PATH}")
print(f"Final shape: {df_clean.shape}")
print(df_clean.dtypes.to_string())
