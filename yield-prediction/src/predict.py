import joblib
import numpy as np

# Load saved components
model = joblib.load("../models/yield_model.pkl")
le = joblib.load("../models/label_encoders.pkl")
scaler = joblib.load("../models/scaler.pkl")

def safe_transform(le, column, value):
    """Safely transform categorical values using a fitted LabelEncoder"""
    if value in le[column].classes_:
        return le[column].transform([value])[0]
    else:
        print(f"⚠️ Warning: '{value}' not seen in training for column '{column}'. Defaulting to first class.")
        return le[column].transform([le[column].classes_[0]])[0]

def predict_yield(crop, season, state, year, area, production, rainfall, fertilizer, pesticide):
    # Encode categorical features safely
    crop_encoded = safe_transform(le, "Crop", crop)
    season_encoded = safe_transform(le, "Season", season)
    state_encoded = safe_transform(le, "State", state)

    # Feature vector
    features = np.array([[crop_encoded, year, season_encoded, state_encoded,
                          area, production, rainfall, fertilizer, pesticide]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)
    return prediction[0]

# Example run
if __name__ == "__main__":
    result = predict_yield("Rice", "Kharif", "Kerala", 2023, 2000, 4000, 1200, 500, 50)
    print("🌾 Predicted Yield:", result)
