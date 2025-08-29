import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Preprocess dataset:
    - Label encode categorical columns
    - Scale numerical columns
    """
    # Encode categorical columns
    label_cols = ["Crop", "Season", "State"]
    le = {}
    for col in label_cols:
        le[col] = LabelEncoder()
        df[col] = le[col].fit_transform(df[col].astype(str))

    # Split features & target
    X = df.drop("Yield", axis=1)
    y = df["Yield"]

    # Scale numerical columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, scaler
