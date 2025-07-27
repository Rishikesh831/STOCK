import numpy as np
import pandas as pd
from src.utils.model_utils import load_model_and_scaler
from src.config import FEATURES_TO_SCALE, WINDOW_SIZE

def create_windows(df, features=FEATURES_TO_SCALE, window_size=WINDOW_SIZE):
    X = []
    meta_rows = []
    for stock in df['Stock'].unique():
        stock_data = df[df['Stock'] == stock]
        feature_matrix = stock_data[features].values
        for i in range(len(stock_data) - window_size):
            X.append(feature_matrix[i:i+window_size])
            meta_rows.append(stock_data.iloc[i+window_size].to_dict())
    X = np.array(X)
    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df

def predict(df):
    model, scaler = load_model_and_scaler()
    scaled_df = df.copy()
    scaled_df[[f for f in FEATURES_TO_SCALE if f != "LogReturn"]] = scaler.transform(scaled_df[[f for f in FEATURES_TO_SCALE if f != "LogReturn"]])
    X_pred, meta_df = create_windows(scaled_df)
    if X_pred.shape[0] == 0:
        raise ValueError("Not enough data for prediction windows.")
    y_pred_log = model.predict(X_pred, verbose=0).flatten()
    return meta_df, y_pred_log 