import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

# Paths
DATA_PATH = Path("../data/mainDataset.csv")
MODEL_PATH = Path("../models/yield_model.pkl")
PREPROCESSOR_PATH = Path("../models/preprocessor.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and target
features = ["Crop", "Season", "State", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]
target = "Yield"

X = df[features]
y = df[target]

# Define categorical & numeric columns
categorical_cols = ["Crop", "Season", "State"]
numeric_cols = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]

# Preprocessor: OneHot for categoricals, StandardScaler for numerics
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# Pipeline: preprocessing + model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Training complete")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Save model (pipeline includes preprocessing!)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
