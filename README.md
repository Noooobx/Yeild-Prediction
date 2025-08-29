#  Crop Yield Prediction System

A Machine Learning project to predict **crop yield** based on various agricultural, climatic, and input parameters.  
This project is built using **Python (scikit-learn, pandas, numpy)** and trained on a dataset containing crop, area, production, rainfall, fertilizer, pesticide, and seasonal details.

---

##  Features
- Predicts crop yield using:
  - Crop type
  - Season
  - State
  - Year
  - Area (hectares)
  - Annual Rainfall (mm)
  - Fertilizer usage (kg)
  - Pesticide usage (kg)
- Machine Learning pipeline:
  - Data preprocessing (encoding, scaling)
  - Training & evaluation
  - Model prediction for user input
- Easily extendable to include **soil data via APIs** (e.g., SoilGrids, ISRO Bhuvan, Gemini API).

---

##  Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn (for visualization)  
- **Algorithms:** Random Forest Regressor (default), Linear Regression, XGBoost (optional)  
- **Future Scope:** Flask/Streamlit integration for a web UI  

---

##  Dataset
The dataset used has the following columns:

- `Crop` → Crop name (categorical)  
- `Crop_Year` → Year of cultivation  
- `Season` → Season (Kharif, Rabi, etc.)  
- `State` → State in India  
- `Area` → Cultivated area (in hectares)  
- `Production` → Production (in tonnes)  
- `Annual_Rainfall` → Rainfall in mm  
- `Fertilizer` → Fertilizer used (kg)  
- `Pesticide` → Pesticide used (kg)  
- `Yield` → Yield (Production/Area, target variable)

---

##  How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Noooobx/Yeild-Prediction.git
   cd Yeild-Prediction
