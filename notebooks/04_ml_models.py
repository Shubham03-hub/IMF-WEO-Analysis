
import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree           import DecisionTreeRegressor
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed — will skip XGBoost models. "
          "Run: pip install xgboost")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
FEAT_PATH   = os.path.join(ROOT_DIR, 'data', 'processed', 'weo_features.csv')
MODELS_DIR  = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("Libraries loaded.")

# Load features ─────────────────────────────────────────────────────
if not os.path.exists(FEAT_PATH):
    raise FileNotFoundError(
        f"[ERROR] Run notebook 03 first — file not found:\n  {FEAT_PATH}"
    )

df = pd.read_csv(FEAT_PATH)
print(f"Shape: {df.shape}")

TARGET   = 'GDP_growth_pct'
DROP_COLS = ['COUNTRY', 'YEAR', TARGET]
feature_cols = [c for c in df.columns if c not in DROP_COLS]

X = df[feature_cols]
y = df[TARGET]
print(f"Features: {len(feature_cols)} | Target: {TARGET}")
print(f"Feature list: {feature_cols}")

# ── Time-aware train / test split ─────────────────────────────────────
# Using years < 2020 for training, 2020+ for testing.
# This avoids data leakage that would occur with a random split.
train_mask = df['YEAR'] < 2020
X_train, X_test = X[train_mask],  X[~train_mask]
y_train, y_test = y[train_mask],  y[~train_mask]
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

if len(X_train) < 10 or len(X_test) < 5:
    raise ValueError(
        "[ERROR] Insufficient data for train/test split. "
        "Check that your dataset covers multiple years including 2020+."
    )

#  Define models ─────────────────────────────────────────────────────
models = {
    'Linear Regression':  LinearRegression(),
    'Ridge Regression':   Ridge(alpha=1.0),
    'Decision Tree':      DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
}
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBRegressor(
        n_estimators=100, random_state=42, verbosity=0, n_jobs=-1
    )

# Train and evaluate all models ─────────────────────────────────────
results       = []
trained_models = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2   = r2_score(y_test, y_pred)

        results.append({'Model': name, 'MAE': round(mae, 4),
                        'RMSE': round(rmse, 4), 'R2': round(r2, 4)})
        trained_models[name] = model
        print(f"{name:<25}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
results_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)
print("\nModel comparison saved.")
print(results_df.to_string(index=False))

# Hyperparameter tuning — best model ───────────────────────────────
# Use the model with the highest R² for tuning.
best_name = results_df.iloc[0]['Model']
print(f"\nTuning: {best_name}")

if best_name == 'XGBoost' and XGBOOST_AVAILABLE:
    param_grid = {
        'n_estimators':  [100, 200],
        'max_depth':     [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
    }
    base = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
elif best_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth':    [None, 5, 10],
        'min_samples_split': [2, 5],
    }
    base = RandomForestRegressor(random_state=42, n_jobs=-1)
elif best_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators':  [100, 200],
        'max_depth':     [3, 5],
        'learning_rate': [0.05, 0.1],
    }
    base = GradientBoostingRegressor(random_state=42)
else:
    # Fallback: no tuning needed for linear models; just use top model as-is
    param_grid = None
    base       = None

if param_grid and base is not None:
    grid_search = GridSearchCV(
        base, param_grid,
        cv=min(5, len(X_train) // 10),   # guard against tiny datasets
        scoring='r2', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params : {grid_search.best_params_}")
    print(f"Best CV R²  : {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_
else:
    best_model = trained_models[best_name]
    print(f"No tuning required — using {best_name} as-is.")

#  Final evaluation of best model ───────────────────────────────────
y_pred_best = best_model.predict(X_test)
print(f"\n{best_name} — Test Set Results:")
print(f"  MAE  : {mean_absolute_error(y_test, y_pred_best):.4f}")
print(f"  RMSE : {mean_squared_error(y_test, y_pred_best)**0.5:.4f}")
print(f"  R²   : {r2_score(y_test, y_pred_best):.4f}")

# ── Cell 8: Save best model and feature list ──────────────────────────────────
joblib.dump(best_model,   os.path.join(MODELS_DIR, 'best_model.pkl'))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.pkl'))
print(f"\n✅ Saved: models/best_model.pkl")
print(f"✅ Saved: models/feature_cols.pkl")
