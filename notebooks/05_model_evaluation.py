# Imports ───────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
FEAT_PATH  = os.path.join(ROOT_DIR, 'data', 'processed', 'weo_features.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SAVE_DIR   = os.path.join(ROOT_DIR, 'reports', 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model and data ───────────────────────────────────────────────
for path, label in [(FEAT_PATH,       'weo_features.csv (run notebook 03)'),
                    (os.path.join(MODELS_DIR, 'best_model.pkl'),   'best_model.pkl (run notebook 04)'),
                    (os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'feature_cols.pkl (run notebook 04)')]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Missing: {path}\n  Please run: {label}")

df          = pd.read_csv(FEAT_PATH)
best_model  = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))

TARGET    = 'GDP_growth_pct'
test_mask = df['YEAR'] >= 2020

# Keep only feature columns that actually exist in the loaded dataframe
feature_cols = [c for c in feature_cols if c in df.columns]

X_test = df.loc[test_mask, feature_cols]
y_test = df.loc[test_mask, TARGET]
y_pred = best_model.predict(X_test)

print(f"Test set size: {len(y_test):,}")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")

# Actual vs Predicted plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue')
lim = [min(y_test.min(), y_pred.min()) - 1,
       max(y_test.max(), y_pred.max()) + 1]
ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual GDP Growth (%)')
ax.set_ylabel('Predicted GDP Growth (%)')
ax.set_title('Actual vs Predicted GDP Growth — Test Set (2020+)', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: actual_vs_predicted.png")

# Feature importance ────────────────────────────────────────────────
# Works for tree-based models; skips gracefully for linear models.
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('Blues_r', len(top_features))
    top_features.plot(kind='barh', ax=ax, color=colors)
    ax.invert_yaxis()
    ax.set_title('Top 12 Feature Importances — Best Model', fontsize=13)
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: feature_importance.png")
    print("\nTop features:")
    print(top_features.to_string())
elif hasattr(best_model, 'coef_'):
    coefs = pd.Series(np.abs(best_model.coef_), index=feature_cols)
    top_features = coefs.sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features.plot(kind='barh', ax=ax, color='steelblue')
    ax.invert_yaxis()
    ax.set_title('Top 12 Feature Coefficients (|value|) — Linear Model', fontsize=13)
    ax.set_xlabel('|Coefficient|')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: feature_importance.png (coefficients)")
else:
    print("[SKIP] Model has no feature_importances_ or coef_ attribute.")

# Residuals plot ────────────────────────────────────────────────────
residuals = np.array(y_test) - np.array(y_pred)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Residuals vs Fitted
axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.2)
axes[0].set_title('Residuals vs Fitted')
axes[0].set_xlabel('Predicted GDP Growth (%)')
axes[0].set_ylabel('Residuals')

# Right: Residuals distribution
axes[1].hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--', linewidth=1.2)
axes[1].set_title('Residuals Distribution')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Count')

plt.suptitle('Residual Analysis', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'residuals.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: residuals.png")

# Model comparison table ───────────────────────────────────────────
comp_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
if os.path.exists(comp_path):
    results_df = pd.read_csv(comp_path)
    print("\nAll model results:")
    print(results_df.to_string(index=False))

    # Styled display (works in Jupyter; plain print in script mode)
    try:
        display(
            results_df.style
            .background_gradient(subset=['R2'],        cmap='Greens')
            .background_gradient(subset=['MAE','RMSE'], cmap='Reds_r')
            .format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R2': '{:.4f}'})
        )
    except NameError:
        pass  # Not in a Jupyter kernel — plain print is fine
else:
    print("[SKIP] model_comparison.csv not found (run notebook 04 first).")

print("\n✅ Notebook 05 complete. Evaluation figures saved to reports/figures/")
