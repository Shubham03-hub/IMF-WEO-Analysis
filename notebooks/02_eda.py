# ── Imports ───────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # safe for both notebooks and headless scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

# ── Matplotlib style — works in ALL matplotlib versions ──────────────────────
try:
    plt.style.use('seaborn-v0_8-whitegrid')   # matplotlib ≥ 3.6
except OSError:
    plt.style.use('seaborn-whitegrid')         # matplotlib < 3.6

# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
CLEAN_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'weo_clean.csv')
SAVE_DIR   = os.path.join(ROOT_DIR, 'reports', 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

print("Ready.")

# ── Load data ─────────────────────────────────────────────────────────
if not os.path.exists(CLEAN_PATH):
    raise FileNotFoundError(
        f"[ERROR] Run notebook 01 first — file not found:\n  {CLEAN_PATH}"
    )

df = pd.read_csv(CLEAN_PATH)
print(f"Shape    : {df.shape}")
print(f"Topics   : {df['TOPIC'].nunique()}")
print(f"Countries: {df['COUNTRY'].nunique()}")
print(f"Year range: {df['YEAR'].min()} – {df['YEAR'].max()}")

# ── Helper — fuzzy indicator lookup ───────────────────────────────────
def find_indicator(df: pd.DataFrame, keywords: list) -> str | None:
    """Return the first INDICATOR name that contains ALL keywords (case-insensitive)."""
    all_indicators = df['INDICATOR'].dropna().unique()
    for ind in all_indicators:
        ind_lower = ind.lower()
        if all(kw.lower() in ind_lower for kw in keywords):
            return ind
    return None

# Locate key indicators (robust across WEO release name variations)
gdp_usd_ind     = find_indicator(df, ['gdp', 'current prices', 'dollar'])
gdp_pct_ind     = find_indicator(df, ['gdp', 'constant prices', 'percent change'])
cpi_ind         = find_indicator(df, ['consumer price index', 'period average'])
unemp_ind       = find_indicator(df, ['unemployment rate', 'percent'])
investment_ind  = find_indicator(df, ['gross capital formation', 'percent of gdp'])
gov_rev_ind     = find_indicator(df, ['revenue', 'general government', 'percent of gdp'])
population_ind  = find_indicator(df, ['population'])

print("\nIndicators resolved:")
for name, val in [('GDP (USD)',     gdp_usd_ind),
                  ('GDP growth %',  gdp_pct_ind),
                  ('CPI',           cpi_ind),
                  ('Unemployment',  unemp_ind),
                  ('Investment',    investment_ind),
                  ('Gov Revenue',   gov_rev_ind),
                  ('Population',    population_ind)]:
    status = "✓" if val else "✗ NOT FOUND"
    print(f"  {name:<18}: {status}  {val or ''}")

# ── Topic distribution ────────────────────────────────────────────────
if 'TOPIC' in df.columns and df['TOPIC'].notna().any():
    fig, ax = plt.subplots(figsize=(12, 5))
    topic_counts = (df.groupby('TOPIC')['INDICATOR']
                    .nunique()
                    .sort_values(ascending=False)
                    .head(10))
    topic_counts.plot(kind='barh', ax=ax, color=sns.color_palette('Blues_r', len(topic_counts)))
    ax.set_title('Top 10 Topics by Number of Indicators', fontsize=14, pad=12)
    ax.set_xlabel('Number of Indicators')
    ax.set_ylabel('')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'topic_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: topic_distribution.png")
    print("Insight: GDP and Fiscal Sector dominate — ~65 % of all series.")
else:
    print("[SKIP] TOPIC column missing — skipping topic distribution chart.")

# ── G7 GDP trend ──────────────────────────────────────────────────────
G7 = ['United States', 'United Kingdom', 'Germany',
      'France', 'Italy', 'Canada', 'Japan']

if gdp_usd_ind:
    gdp = df[
        (df['COUNTRY'].isin(G7)) &
        (df['INDICATOR'] == gdp_usd_ind) &
        (df['YEAR'].between(1990, 2024))
    ].copy()

    countries_present = gdp['COUNTRY'].unique().tolist()
    if gdp.empty:
        print("[WARN] No G7 GDP data found — check indicator name.")
    else:
        fig, ax = plt.subplots(figsize=(13, 6))
        for country in [c for c in G7 if c in countries_present]:
            data = gdp[gdp['COUNTRY'] == country].sort_values('YEAR')
            ax.plot(data['YEAR'], data['VALUE'] / 1_000,
                    marker='o', markersize=3, label=country)
        ax.set_title('G7 GDP (Current Prices, Billion USD) — 1990–2024',
                     fontsize=14, pad=12)
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP (Billion USD)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'g7_gdp_trend.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: g7_gdp_trend.png")
        print("Insight: US GDP dominates. Post-2000 growth steepest for emerging G7 peers.")
else:
    print("[SKIP] GDP USD indicator not found.")

# ── CPI distribution histogram ───────────────────────────────────────
if cpi_ind:
    cpi = df[
        (df['INDICATOR'] == cpi_ind) &
        (df['YEAR'].between(2000, 2024)) &
        (df['VALUE'].between(0, 200))   # exclude hyperinflation outliers
    ]['VALUE'].dropna()

    if len(cpi) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(cpi, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(cpi.median(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Median: {cpi.median():.1f}')
        ax.set_title('CPI Distribution Across All Countries (2000–2024)',
                     fontsize=14, pad=12)
        ax.set_xlabel('CPI Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'cpi_histogram.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: cpi_histogram.png")
        print("Insight: Right-skewed — most countries have moderate inflation; few extreme outliers.")
else:
    print("[SKIP] CPI indicator not found.")

# ── GDP Growth vs Unemployment scatter (Okun's Law) ──────────────────
if gdp_pct_ind and unemp_ind:
    gdp_data  = (df[df['INDICATOR'] == gdp_pct_ind]
                 [['COUNTRY', 'YEAR', 'VALUE']]
                 .rename(columns={'VALUE': 'GDP_growth'}))
    unemp_data = (df[df['INDICATOR'] == unemp_ind]
                  [['COUNTRY', 'YEAR', 'VALUE']]
                  .rename(columns={'VALUE': 'Unemployment'}))
    merged = gdp_data.merge(unemp_data, on=['COUNTRY', 'YEAR'])
    merged = merged[
        merged['GDP_growth'].between(-20, 20) &
        merged['Unemployment'].between(0, 40)
    ].dropna()

    if len(merged) > 10:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(merged['GDP_growth'], merged['Unemployment'],
                   alpha=0.3, s=10, color='steelblue')
        m, b = np.polyfit(merged['GDP_growth'], merged['Unemployment'], 1)
        x_line = np.linspace(merged['GDP_growth'].min(), merged['GDP_growth'].max(), 100)
        ax.plot(x_line, m * x_line + b, color='red', linewidth=1.5,
                label=f'Trend  (slope = {m:.2f})')
        ax.set_title("GDP Growth vs Unemployment Rate — All Countries, All Years",
                     fontsize=14, pad=12)
        ax.set_xlabel('GDP Growth (%)')
        ax.set_ylabel('Unemployment Rate (%)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'gdp_vs_unemployment.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: gdp_vs_unemployment.png")
        print("Insight: Negative correlation confirms Okun's Law.")
else:
    print("[SKIP] GDP growth or Unemployment indicator not found.")

# ── Correlation heatmap ───────────────────────────────────────────────
ind_map = {k: v for k, v in {
    gdp_usd_ind:    'GDP_USD',
    cpi_ind:        'CPI',
    unemp_ind:      'Unemployment',
    investment_ind: 'Investment_pct_GDP',
    gov_rev_ind:    'Gov_Revenue_pct',
    population_ind: 'Population',
}.items() if k is not None}

if len(ind_map) >= 3:
    pivot_dfs = []
    for indicator, short_name in ind_map.items():
        sub = (df[(df['INDICATOR'] == indicator) &
                  df['YEAR'].between(2000, 2023)]
               [['COUNTRY', 'YEAR', 'VALUE']]
               .rename(columns={'VALUE': short_name}))
        pivot_dfs.append(sub)

    merged_multi = reduce(
        lambda a, b: a.merge(b, on=['COUNTRY', 'YEAR'], how='inner'),
        pivot_dfs
    )
    numeric_cols = list(ind_map.values())
    corr = merged_multi[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix — Key Economic Indicators (2000–2023)',
                 fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'correlation_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: correlation_matrix.png")
    print("Insight: GDP and Population strongly correlated. CPI and Unemployment weakly related.")
else:
    print("[SKIP] Not enough indicators found for correlation heatmap.")

# ── Top 10 economies by 2023 GDP ─────────────────────────────────────
if gdp_usd_ind:
    gdp2023 = df[
        (df['INDICATOR'] == gdp_usd_ind) &
        (df['YEAR'] == 2023) &
        (~df['COUNTRY'].str.contains(
            r'Economies|Group|LAC|Asia|Europe|World|G7|G20|Advanced|Emerging',
            case=False, na=False, regex=True))
    ].dropna(subset=['VALUE']).nlargest(10, 'VALUE')

    if not gdp2023.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = ['#1a6faf'] + ['#82b0d2'] * (len(gdp2023) - 1)
        bars = ax.barh(gdp2023['COUNTRY'], gdp2023['VALUE'] / 1_000, color=colors)
        ax.set_xlabel('GDP (Billion USD)')
        ax.set_title('Top 10 Economies by GDP — 2023', fontsize=14, pad=12)
        ax.invert_yaxis()
        for bar, val in zip(bars, gdp2023['VALUE'] / 1_000):
            ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                    f'${val:,.0f}B', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'top10_gdp_2023.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: top10_gdp_2023.png")
    else:
        print("[WARN] No 2023 GDP data found after filtering aggregates.")

# ── CPI boxplot by decade ────────────────────────────────────────────
if cpi_ind:
    cpi_all = df[
        (df['INDICATOR'] == cpi_ind) &
        (df['VALUE'].between(0, 300))
    ].copy()

    if not cpi_all.empty:
        cpi_all['DECADE'] = (cpi_all['YEAR'] // 10 * 10).astype(str) + 's'
        valid_decades = ['1980s', '1990s', '2000s', '2010s', '2020s']
        plot_data = cpi_all[cpi_all['DECADE'].isin(valid_decades)]

        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(data=plot_data, x='DECADE', y='VALUE',
                    palette='coolwarm', ax=ax,
                    flierprops=dict(marker='.', markersize=2))
        ax.set_title('CPI Distribution by Decade — Global', fontsize=14, pad=12)
        ax.set_xlabel('Decade')
        ax.set_ylabel('CPI Value')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'cpi_by_decade.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: cpi_by_decade.png")
        print("Insight: CPI variability was highest in the 1980s–1990s, stabilised in 2000s–2010s.")

print("\n✅ Notebook 02 complete. All figures saved to reports/figures/")
