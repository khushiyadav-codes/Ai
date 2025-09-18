import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Housing Price Predictor", layout="wide")
st.title("🏠 AI Housing Price Predictor")
st.write("Interactive demo using the California Housing dataset (built with Streamlit + scikit-learn).")

# --- Load data (no caching to avoid pickling/hash errors on Cloud)
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
feature_names = X.columns.tolist()

# --- Train models (simple & fast; no caching)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=150, random_state=42).fit(X_train, y_train)

def eval_model(model, name):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred),
    }

results = pd.DataFrame([eval_model(lr, "Linear Regression"),
                        eval_model(rf, "Random Forest")])

st.subheader("Model Performance")
st.dataframe(results.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}))

# --- What-If panel
st.subheader("Try 'What-If' Prediction")
col1, col2, col3 = st.columns(3)
defaults = X.mean()

with col1:
    MedInc = st.slider("Median Income (10k$)", float(X["MedInc"].min()), float(X["MedInc"].max()), float(defaults["MedInc"]))
    HouseAge = st.slider("House Age (years)", float(X["HouseAge"].min()), float(X["HouseAge"].max()), float(defaults["HouseAge"]))
    AveRooms = st.slider("Avg Rooms", float(X["AveRooms"].min()), float(X["AveRooms"].max()), float(defaults["AveRooms"]))
with col2:
    AveBedrms = st.slider("Avg Bedrooms", float(X["AveBedrms"].min()), float(X["AveBedrms"].max()), float(defaults["AveBedrms"]))
    Population = st.slider("Population", float(X["Population"].min()), float(X["Population"].max()), float(defaults["Population"]))
    AveOccup = st.slider("Avg Occupancy", float(X["AveOccup"].min()), float(X["AveOccup"].max()), float(defaults["AveOccup"]))
with col3:
    Latitude = st.slider("Latitude", float(X["Latitude"].min()), float(X["Latitude"].max()), float(defaults["Latitude"]))
    Longitude = st.slider("Longitude", float(X["Longitude"].min()), float(X["Longitude"].max()), float(defaults["Longitude"]))

row = {
    "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms, "AveBedrms": AveBedrms,
    "Population": Population, "AveOccup": AveOccup, "Latitude": Latitude, "Longitude": Longitude
}
X_in = pd.DataFrame([row])[feature_names]

model_choice = st.selectbox("Model for prediction", ["Random Forest", "Linear Regression"])
model = rf if model_choice == "Random Forest" else lr
pred = model.predict(X_in)[0]
st.metric("Predicted Median House Value (×100k USD)", f"{pred:.2f}")

# --- Quick EDA (tiny sample to keep it light)
st.subheader("Quick EDA")
st.write("Sample of the dataset:")
st.dataframe(pd.concat([X, y.rename("MedHouseVal")], axis=1).head(10))
