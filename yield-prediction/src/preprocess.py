import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Preprocess dataset:
    - Handle missing values
    - Label encode categorical columns
    - Scale numerical columns
    - Exclude Crop_Year
    """
    df = df.dropna()  # or handle NaNs properly
    
    label_cols = ["Crop", "Season", "State"]
    le = {}
    for col in label_cols:
        le[col] = LabelEncoder()
        df[col] = le[col].fit_transform(df[col].astype(str))

    features = ["Crop", "Season", "State", "Area", "Production", 
                "Annual_Rainfall", "Fertilizer", "Pesticide"]
    X = df[features]
    y = df["Yield"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame with same column names
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)

    return X_scaled, y, le, scaler
