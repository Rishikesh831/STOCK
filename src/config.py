import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'log_return_lstm_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

# Features and window
FEATURES_TO_SCALE = [
    "LogReturn",
    "Vol_10", "Mom_10",
    "Vol_20", "Mom_20",
    "Vol_60", "Mom_60"
]
WINDOW_SIZE = 60 