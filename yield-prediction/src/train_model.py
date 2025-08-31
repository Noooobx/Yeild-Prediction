import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("../data/mainDataset.csv")

# Preprocess
X, y, le, scaler = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Training Complete")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model & preprocessing objects
joblib.dump(model, "../models/yield_model.pkl")
joblib.dump(le, "../mas iw as doing knowleodels/label_encoders.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
print("✅ Model and encoders saved in /models/")
