import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Preprocess dataset:
    - Label encode categorical columns
    - Scale numerical columns
    - Exclude Crop_Year (future-ready)
    """
    # Encode categorical columns
    label_cols = ["Crop", "Season", "State"]
    le = {}
    for col in label_cols:
        le[col] = LabelEncoder()
        df[col] = le[col].fit_transform(df[col].astype(str))

    # Select features (exclude Crop_Year)
    features = ["Crop", "Season", "State", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]
    X = df[features]
    y = df["Yield"]

    print("Model features:", X.columns.tolist())

    # Scale numerical columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, scaler
