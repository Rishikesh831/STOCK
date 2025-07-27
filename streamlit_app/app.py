import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to include parent directory for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.preprocess import preprocess_input
from src.pipeline.predict import predict
from src.pipeline.postprocess import format_results
from src.utils.file_utils import load_input_file

# Page config
st.set_page_config(page_title="📊 Stock Return Predictor", layout="wide", page_icon="📈")

# Title and Intro
st.markdown("<h1 style='text-align: center;'>📈 Stock Return Predictor Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Author: <b>Rishikesh Bhatt</b></div><br>", unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.header("📂 Upload Stock Data")
    file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

# When file is uploaded
if file:
    try:
        df = load_input_file(file)
    except Exception as e:
        st.error(f"❌ File loading error: {e}")
        st.stop()

    try:
        df_proc = preprocess_input(df)
    except Exception as e:
        st.error(f"❌ Preprocessing error: {e}")
        st.stop()

    try:
        meta_df, y_pred_log = predict(df_proc)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    try:
        results = format_results(meta_df, y_pred_log)
    except Exception as e:
        st.error(f"❌ Postprocessing error: {e}")
        st.stop()

    # Styling max/min values
    def highlight_max_min(s):
        max_idx = s.idxmax()
        min_idx = s.idxmin()
        return [
            "background-color: #c6f6d5; font-weight: bold;" if i == max_idx else
            ("background-color: #fed7d7; font-weight: bold;" if i == min_idx else "")
            for i in s.index
        ]

    # Display results
    st.subheader("📋 Predicted Returns Table")
    styled_df = results[["Date", "Stock", "Predicted Return (%)"]].style.apply(
        highlight_max_min, subset=["Predicted Return (%)"]
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Line Chart for predicted returns
    st.subheader("📉 Predicted Returns Over Time")
    chart_df = results[["Date", "Predicted Return (%)"]].copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
    chart_df = chart_df.sort_values("Date")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.line_chart(chart_df.set_index("Date"))
    with col2:
        st.metric(label="📈 Max Return (%)", value=f"{chart_df['Predicted Return (%)'].max():.2f}")
        st.metric(label="📉 Min Return (%)", value=f"{chart_df['Predicted Return (%)'].min():.2f}")

    # Download predictions
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="predicted_returns.csv",
        mime="text/csv"
    )

else:
    st.info("📁 Upload a CSV or Excel file with columns: Date, Stock, Open, High, Low, Close, Volume.")
