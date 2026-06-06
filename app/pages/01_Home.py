
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from utils.load_data import load_clean_data

st.set_page_config(page_title="Dataset Overview", page_icon="📊", layout="wide")
st.title("📊 Dataset Overview")

df = load_clean_data()

# ── KPI metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows",    f"{len(df):,}")
col2.metric("Countries",      df['COUNTRY'].nunique())
col3.metric("Indicators",     df['INDICATOR'].nunique())
if 'TOPIC' in df.columns:
    col4.metric("Topics", df['TOPIC'].nunique())
else:
    col4.metric("Year Range", f"{int(df['YEAR'].min())}–{int(df['YEAR'].max())}")

# ── Topic breakdown ───────────────────────────────────────────────────────────
if 'TOPIC' in df.columns and df['TOPIC'].notna().any():
    st.subheader("Topic Breakdown")
    topic_counts = (df.groupby('TOPIC')['INDICATOR']
                    .nunique()
                    .sort_values(ascending=False))
    st.bar_chart(topic_counts.head(8))
else:
    st.subheader("Top Indicators by Coverage")
    ind_counts = df.groupby('INDICATOR')['COUNTRY'].nunique().sort_values(ascending=False)
    st.bar_chart(ind_counts.head(10))

# ── Data summary ──────────────────────────────────────────────────────────────
st.subheader("Missing Values by Column")
missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
st.dataframe(
    missing_pct[missing_pct > 0]
    .sort_values(ascending=False)
    .rename("Missing %")
    .reset_index()
    .rename(columns={"index": "Column"}),
    use_container_width=True
)

# ── Sample rows ───────────────────────────────────────────────────────────────
st.subheader("Sample Data")
st.dataframe(df.sample(min(20, len(df))), use_container_width=True)
