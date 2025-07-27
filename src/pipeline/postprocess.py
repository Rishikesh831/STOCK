import numpy as np
import pandas as pd

def logreturn_to_return(log_returns):
    return np.exp(log_returns) - 1

def format_results(meta_df, y_pred):
    results = meta_df.copy()
    results["Predicted Return"] = logreturn_to_return(y_pred)
    results["Predicted Return (%)"] = results["Predicted Return"] * 100
    results = results.sort_values("Predicted Return", ascending=False).reset_index(drop=True)
    return results 