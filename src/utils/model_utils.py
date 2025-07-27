import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
from src.config import MODEL_PATH, SCALER_PATH

_model = None
_scaler = None

def load_model_and_scaler():
    global _model, _scaler
    if _model is None:
        _model = load_model(MODEL_PATH, custom_objects={'Huber': Huber()})
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler 