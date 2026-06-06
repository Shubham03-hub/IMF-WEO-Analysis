
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.load_data import load_clean_data, get_countries, get_indicators

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")
st.title("🔍 Exploratory Data Analysis")

df = load_clean_data()
all_countries  = get_countries(df)
all_indicators = get_indicators(df)

# ── Helper: fuzzy indicator lookup ───────────────────────────────────────────
def find_indicator(keywords: list) -> str | None:
    for ind in all_indicators:
        if all(kw.lower() in ind.lower() for kw in keywords):
            return ind
    return None

gdp_usd_ind = find_indicator(['gdp', 'current prices', 'dollar'])
gdp_pct_ind = find_indicator(['gdp', 'constant prices', 'percent change'])
cpi_ind     = find_indicator(['consumer price index', 'period average'])
unemp_ind   = find_indicator(['unemployment rate', 'percent'])

# ── Section 1: GDP Trend ──────────────────────────────────────────────────────
st.subheader("GDP Trend by Country")

default_countries = [c for c in ['United States', 'China', 'Germany', 'India']
                     if c in all_countries]
countries_sel = st.multiselect(
    "Select countries",
    options=all_countries,
    default=default_countries or all_countries[:4],
)

year_range = st.slider("Year range", 1990, 2024, (2000, 2023))

if gdp_usd_ind:
    filtered = df[
        df['COUNTRY'].isin(countries_sel) &
        (df['INDICATOR'] == gdp_usd_ind) &
        df['YEAR'].between(*year_range)
    ].copy()
    filtered['GDP_Billion_USD'] = filtered['VALUE'] / 1_000

    if not filtered.empty:
        fig = px.line(
            filtered.sort_values('YEAR'),
            x='YEAR', y='GDP_Billion_USD', color='COUNTRY',
            title='GDP — Current Prices (Billion USD)',
            labels={'GDP_Billion_USD': 'GDP (Billion USD)', 'YEAR': 'Year'},
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data for selected filters. Try different countries or years.")
else:
    st.warning("GDP (Current Prices, USD) indicator not found in dataset.")

# ── Section 2: GDP Growth % ────────────────────────────────────────────────────
if gdp_pct_ind:
    st.subheader("GDP Growth Rate (%)")
    growth_countries = st.multiselect(
        "Select countries (growth)",
        options=all_countries,
        default=default_countries or all_countries[:4],
        key="growth_countries",
    )
    filtered_growth = df[
        df['COUNTRY'].isin(growth_countries) &
        (df['INDICATOR'] == gdp_pct_ind) &
        df['YEAR'].between(*year_range)
    ]
    if not filtered_growth.empty:
        fig2 = px.line(
            filtered_growth.sort_values('YEAR'),
            x='YEAR', y='VALUE', color='COUNTRY',
            title='GDP Growth Rate (Constant Prices, % Change)',
            labels={'VALUE': 'GDP Growth (%)', 'YEAR': 'Year'},
        )
        fig2.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

# ── Section 3: Indicator Explorer ─────────────────────────────────────────────
st.subheader("Indicator Explorer")
col_a, col_b = st.columns(2)
with col_a:
    indicator_sel = st.selectbox("Select indicator", all_indicators)
with col_b:
    country_sel = st.selectbox("Select country", all_countries)

ind_data = df[
    (df['INDICATOR'] == indicator_sel) &
    (df['COUNTRY']   == country_sel)
].dropna(subset=['VALUE']).sort_values('YEAR')

if not ind_data.empty:
    fig3 = px.bar(
        ind_data, x='YEAR', y='VALUE',
        title=f"{indicator_sel} — {country_sel}",
        labels={'VALUE': 'Value', 'YEAR': 'Year'},
        color='VALUE',
        color_continuous_scale='Blues',
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No data available for this indicator/country combination.")

# ── Section 4: CPI vs Unemployment scatter ────────────────────────────────────
if cpi_ind and unemp_ind:
    st.subheader("Inflation vs Unemployment (Phillips Curve)")
    year_sel = st.slider("Select year", int(df['YEAR'].min()), 2023, 2019,
                         key="phillips_year")
    cpi_df   = df[(df['INDICATOR'] == cpi_ind)   & (df['YEAR'] == year_sel)][['COUNTRY','VALUE']].rename(columns={'VALUE':'CPI'})
    unemp_df = df[(df['INDICATOR'] == unemp_ind) & (df['YEAR'] == year_sel)][['COUNTRY','VALUE']].rename(columns={'VALUE':'Unemployment'})
    scatter_df = cpi_df.merge(unemp_df, on='COUNTRY').dropna()
    scatter_df = scatter_df[scatter_df['CPI'].between(0, 250) & scatter_df['Unemployment'].between(0, 40)]

    if not scatter_df.empty:
        fig4 = px.scatter(
            scatter_df, x='Unemployment', y='CPI',
            hover_name='COUNTRY', opacity=0.7,
            title=f'CPI vs Unemployment Rate — {year_sel}',
            labels={'CPI': 'CPI (Inflation Index)', 'Unemployment': 'Unemployment (%)'},
            trendline='ols',
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info(f"No cross-country data found for {year_sel}.")
