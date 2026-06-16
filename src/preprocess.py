import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """Load California Housing dataset as a DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print(f"[INFO] Dataset loaded — shape: {df.shape}")
    return df


def remove_outliers(df: pd.DataFrame, column: str, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Remove outliers beyond the given quantile range."""
    q_low  = df[column].quantile(lower)
    q_high = df[column].quantile(upper)
    filtered = df[(df[column] >= q_low) & (df[column] <= q_high)]
    print(f"[INFO] Outlier removal: {len(df) - len(filtered)} rows removed from '{column}'")
    return filtered


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns."""
    df = df.copy()
    df['RoomsPerHousehold']  = df['AveRooms']   / df['AveOccup'].replace(0, np.nan)
    df['BedroomsPerRoom']    = df['AveBedrms']  / df['AveRooms'].replace(0, np.nan)
    df['PopulationPerHouse'] = df['Population'] / df['HouseAge'].replace(0, np.nan)
    df['IncomePerRoom']      = df['MedInc']      / df['AveRooms'].replace(0, np.nan)

    # Fill any NaN produced by division
    df.fillna(df.median(numeric_only=True), inplace=True)
    print(f"[INFO] Feature engineering done — new shape: {df.shape}")
    return df


def prepare_data(test_size: float = 0.2, random_state: int = 42, scale: bool = True):
   
    df = load_data()
    df = remove_outliers(df, 'MedHouseVal')
    df = engineer_features(df)

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    print(f"[INFO] Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    print("Preprocessing complete.")
