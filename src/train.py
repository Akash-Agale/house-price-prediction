import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from preprocess import prepare_data


def evaluate(name: str, model, X_test, y_test) -> dict:
    """Return a metrics dictionary for a fitted model."""
    preds = model.predict(X_test)
    return {
        "Model" : name,
        "RMSE"  : round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "MAE"   : round(mean_absolute_error(y_test, preds), 4),
        "R2"    : round(r2_score(y_test, preds), 4),
    }


def train_baseline_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train several baseline models and return a comparison DataFrame."""
    baselines = {
        "Linear Regression"  : LinearRegression(),
        "Ridge Regression"   : Ridge(alpha=1.0),
        "Lasso Regression"   : Lasso(alpha=0.01),
        "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost"            : xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }

    records = []
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        metrics = evaluate(name, model, X_test, y_test)
        cv_r2   = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
        metrics["CV_R2"] = round(cv_r2, 4)
        records.append(metrics)
        print(f"  {name:<25} R²={metrics['R2']}  RMSE={metrics['RMSE']}  CV_R²={metrics['CV_R2']}")

    return pd.DataFrame(records).sort_values("R2", ascending=False)


def tune_xgboost(X_train, y_train) -> xgb.XGBRegressor:
    """GridSearchCV tuning for XGBoost."""
    param_grid = {
        "n_estimators"  : [100, 200],
        "max_depth"     : [4, 6],
        "learning_rate" : [0.05, 0.1],
        "subsample"     : [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
    gs = GridSearchCV(xgb_model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"\n[TUNING] Best params : {gs.best_params_}")
    print(f"[TUNING] Best CV R²  : {gs.best_score_:.4f}")
    return gs.best_estimator_


def save_artifacts(model, scaler, output_dir: str = "../models"):
    """Persist the trained model and scaler."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model,  os.path.join(output_dir, "xgboost_best_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    print(f"[SAVE] Artifacts written to '{output_dir}/'")


# ── Main ──

if __name__ == "__main__":
    print("=" * 60)
    print("  House Price Prediction — Training Pipeline")
    print("=" * 60)

    # 1. Prepare data (no scaling for tree-based models)
    X_train, X_test, y_train, y_test, scaler = prepare_data(scale=False)

    # 2. Baseline comparison
    print("\n[BASELINES]")
    results_df = train_baseline_models(X_train, X_test, y_train, y_test)
    print("\n", results_df.to_string(index=False))

    # 3. Tune best model
    print("\n[HYPERPARAMETER TUNING — XGBoost]")
    best_model = tune_xgboost(X_train, y_train)

    # 4. Final evaluation
    final_metrics = evaluate("XGBoost (Tuned)", best_model, X_test, y_test)
    print("\n[FINAL MODEL]")
    for k, v in final_metrics.items():
        print(f"  {k:<8}: {v}")

    # 5. Save
    save_artifacts(best_model, scaler)
    print("\n Training complete!")
