import pandas as pd
import numpy as np

def preprocess_input(df):
    required_cols = ["Date", "Stock", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Stock"] = df["Stock"].astype("category").cat.codes
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    df["LogReturn"] = df.groupby("Stock")["Close"].transform(lambda x: np.log(x / x.shift(1)))
    df = df.dropna(subset=["LogReturn"]).reset_index(drop=True)
    for w in [10, 20, 60]:
        df[f"Vol_{w}"] = df.groupby("Stock")["LogReturn"].transform(lambda x: x.rolling(w).std())
        df[f"Mom_{w}"] = df.groupby("Stock")["LogReturn"].transform(lambda x: x.rolling(w).mean())
    df = df.dropna(subset=["Vol_60", "Mom_60"]).reset_index(drop=True)
    return df 