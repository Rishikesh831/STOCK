import pandas as pd
import numpy as np

# Features used for scaling and model input
FEATURES_TO_SCALE = [
    "LogReturn",
    "Vol_10", "Mom_10",
    "Vol_20", "Mom_20",
    "Vol_60", "Mom_60"
]

WINDOW_SIZE = 60
TARGET_SHIFT = 1

# --- Preprocessing Functions ---
def preprocess_dataframe(df):
    """
    Preprocesses the input DataFrame to add log returns, rolling features, and encodes Stock as category code.
    Returns the processed DataFrame.
    """
    # Ensure columns exist
    required_cols = ["Date", "Stock", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Stock"] = df["Stock"].astype("category").cat.codes
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # Calculate log returns per stock
    df["LogReturn"] = df.groupby("Stock")["Close"].transform(lambda x: np.log(x / x.shift(1)))
    df = df.dropna(subset=["LogReturn"]).reset_index(drop=True)

    # Rolling volatility and momentum
    for w in [10, 20, 60]:
        df[f"Vol_{w}"] = df.groupby("Stock")["LogReturn"].transform(lambda x: x.rolling(w).std())
        df[f"Mom_{w}"] = df.groupby("Stock")["LogReturn"].transform(lambda x: x.rolling(w).mean())

    # Drop rows where rolling features are not available
    df = df.dropna(subset=["Vol_60", "Mom_60"]).reset_index(drop=True)
    return df

def create_prediction_windows(df, features=FEATURES_TO_SCALE, window_size=WINDOW_SIZE):
    """
    Create windowed feature arrays for prediction. Returns X (n_samples, window_size, n_features),
    and a DataFrame with the last row of each window for mapping predictions.
    """
    X = []
    meta_rows = []
    for stock in df['Stock'].unique():
        stock_data = df[df['Stock'] == stock]
        feature_matrix = stock_data[features].values
        for i in range(len(stock_data) - window_size):
            X.append(feature_matrix[i:i+window_size])
            # Save the meta row for the prediction (date, stock, etc.)
            meta_rows.append(stock_data.iloc[i+window_size].to_dict())
    X = np.array(X)
    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df 