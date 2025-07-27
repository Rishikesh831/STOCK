import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
from preprocessing import preprocess_dataframe, create_prediction_windows, FEATURES_TO_SCALE, WINDOW_SIZE

st.set_page_config(page_title="Stock Return Prediction Dashboard", layout="wide")
st.title("📈 Stock Return Prediction Dashboard")
st.markdown("**Author: Rishikesh Bhatt**")

MODEL_PATH = os.path.join("model", "log_return_lstm_model.h5")
SCALER_PATH = os.path.join("model", "scaler.pkl")

@st.cache_resource
def load_model_and_scaler():
    model = load_model(MODEL_PATH, custom_objects={'Huber': Huber()})
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"]) 

if file:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    try:
        df_proc = preprocess_dataframe(df)
    except Exception as e:
        st.error(f"❌ Preprocessing error: {e}")
        st.stop()

    # Scale features (excluding LogReturn, which is not used for prediction)
    scaled_df = df_proc.copy()
    try:
        scaled_df[[f for f in FEATURES_TO_SCALE if f != "LogReturn"]] = scaler.transform(scaled_df[[f for f in FEATURES_TO_SCALE if f != "LogReturn"]])
    except Exception as e:
        st.error(f"❌ Scaling error: {e}")
        st.stop()

    # Windowing for prediction
    try:
        X_pred, meta_df = create_prediction_windows(scaled_df, features=FEATURES_TO_SCALE, window_size=WINDOW_SIZE)
    except Exception as e:
        st.error(f"❌ Windowing error: {e}")
        st.stop()

    if X_pred.shape[0] == 0:
        st.warning("Not enough data after preprocessing to make predictions. Please upload more data.")
        st.stop()

    # Predict log returns
    try:
        y_pred_log = model.predict(X_pred, verbose=0).flatten()
        y_pred = np.exp(y_pred_log) - 1
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    # Prepare results DataFrame
    results = meta_df.copy()
    results["Predicted Return"] = y_pred
    results["Predicted Return (%)"] = results["Predicted Return"] * 100
    results = results.sort_values("Predicted Return", ascending=False).reset_index(drop=True)

    # Color-coding for top 5 returns
    def highlight_top5(s):
        is_top5 = s.rank(method='first', ascending=False) <= 5
        return ["background-color: #d4f7d4; font-weight: bold;" if v else "" for v in is_top5]

    st.subheader("Predicted Returns Table")
    styled = results[["Date", "Stock", "Predicted Return (%)"]].style.apply(highlight_top5, subset=["Predicted Return (%)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Predicted Returns Line Chart")
    chart_df = results[["Date", "Predicted Return (%)"]].copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
    chart_df = chart_df.sort_values("Date")
    st.line_chart(chart_df.set_index("Date"))

    # Download button
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, "predicted_returns.csv", "text/csv")
else:
    st.info("Please upload a CSV or Excel file with columns: Date, Stock, Open, High, Low, Close, Volume.") 