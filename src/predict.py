"""
predict.py
----------
Load the saved model and make predictions on new data.
Author : Akash Agale
GitHub : https://github.com/Akash-Agale
"""

import numpy as np
import pandas as pd
import joblib
import os


MODEL_PATH  = os.path.join(os.path.dirname(__file__), "../models/xgboost_best_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/scaler.pkl")

FEATURE_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
    "RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHouse", "IncomePerRoom",
]


def load_model():
    """Load the saved XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )
    return joblib.load(MODEL_PATH)


def engineer_input(data: dict) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    df = pd.DataFrame([data])
    df['RoomsPerHousehold']  = df['AveRooms']   / df['AveOccup'].replace(0, np.nan)
    df['BedroomsPerRoom']    = df['AveBedrms']  / df['AveRooms'].replace(0, np.nan)
    df['PopulationPerHouse'] = df['Population'] / df['HouseAge'].replace(0, np.nan)
    df['IncomePerRoom']      = df['MedInc']      / df['AveRooms'].replace(0, np.nan)
    df.fillna(0, inplace=True)
    return df[FEATURE_COLUMNS]


def predict(input_data: dict) -> float:
    """
    Predict house price for a single input record.

    Parameters
    ----------
    input_data : dict
        Keys: MedInc, HouseAge, AveRooms, AveBedrms,
              Population, AveOccup, Latitude, Longitude

    Returns
    -------
    float : predicted median house value (in $100k)
    """
    model  = load_model()
    df_in  = engineer_input(input_data)
    price  = model.predict(df_in)[0]
    return round(float(price), 4)


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "MedInc"     : 5.0,
        "HouseAge"   : 20.0,
        "AveRooms"   : 6.0,
        "AveBedrms"  : 1.2,
        "Population" : 1200.0,
        "AveOccup"   : 3.0,
        "Latitude"   : 34.05,
        "Longitude"  : -118.25,
    }

    predicted_price = predict(sample)
    print("=" * 45)
    print("  House Price Prediction — Inference Demo")
    print("=" * 45)
    print("\nInput Features:")
    for k, v in sample.items():
        print(f"  {k:<15}: {v}")
    print(f"\n💰 Predicted Price : ${predicted_price * 100_000:,.0f}")
    print(f"   (Raw output     : {predicted_price} × $100k)")
