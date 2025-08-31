from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("models/yield_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

# Clean encoder classes (remove trailing spaces)
for key in ["State", "Season", "Crop"]:
    encoders[key].classes_ = np.array([cls.strip() for cls in encoders[key].classes_])

# Load scaler
scaler = joblib.load("models/scaler.pkl")

app = FastAPI()

# ✅ Add CORS middleware
origins = [
    "http://localhost:5173",   # React dev server
    "http://127.0.0.1:5173",   # alternative
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 👈 allow only frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema with crop
class CropInput(BaseModel):
    crop: str
    season: str
    state: str
    area: float
    production: float
    annual_rainfall: float
    fertilizer: float
    pesticide: float

@app.get("/info")
def get_info():
    return {
        "crops": list(encoders["Crop"].classes_),
        "seasons": list(encoders["Season"].classes_),
        "states": list(encoders["State"].classes_)
    }

@app.post("/predict")
def predict_yield(data: CropInput):
    try:
        # Clean categorical inputs
        crop = data.crop.strip()
        season = data.season.strip()
        state = data.state.strip()

        # Encode categorical features
        crop_encoded = encoders["Crop"].transform([crop])[0]
        season_encoded = encoders["Season"].transform([season])[0]
        state_encoded = encoders["State"].transform([state])[0]

        # Prepare input array in same order as training
        X = np.array([[crop_encoded,
                       season_encoded,
                       state_encoded,
                       data.area,
                       data.production,
                       data.annual_rainfall,
                       data.fertilizer,
                       data.pesticide]])

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict yield
        prediction = model.predict(X_scaled)[0]
        return {"predicted_yield": float(prediction)}

    except Exception as e:
        return {"error": str(e)}
