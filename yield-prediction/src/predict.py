import joblib
import numpy as np
from pathlib import Path

# Paths
MODEL_PATH = Path("../models/yield_model.pkl")

# Load full pipeline (preprocessor + model)
model = joblib.load(MODEL_PATH)

def predict_yield(crop, season, state, area, production, rainfall, fertilizer, pesticide):
    # Feature vector in same order as training
    features = np.array([{
        "Crop": crop,
        "Season": season,
        "State": state,
        "Area": area,
        "Production": production,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }])

    # Predict
    prediction = model.predict(features)[0]
    return float(prediction)

# Example run
if __name__ == "__main__":
    result = predict_yield("Rice", "Kharif", "Kerala", 2000, 4000, 1200, 500, 50)
    print("🌾 Predicted Yield:", result)
