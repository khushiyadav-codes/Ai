import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Housing Price Predictor", layout="wide")

st.title("🏠 AI Housing Price Predictor")
st.write("Interactive demo using the California Housing dataset (built with Streamlit + scikit-learn).")

# Load data
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    feature_names = X.columns.tolist()
    return X, y, feature_names

X, y, feature_names = load_data()

# Train models (simple, fast)
@st.cache_resource
def train_models(random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=150, random_state=random_state).fit(X_train, y_train)

    def eval_model(model, name):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        return {"name": name, "MAE": mae, "RMSE": rmse, "R2": r2}

    results = pd.DataFrame([
        eval_model(lr, "LinearRegression"),
        eval_model(rf, "RandomForest")
    ])
    return lr, rf, results

lr, rf, results = train_models()
st.subheader("Model Performance")
st.dataframe(results.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}))

# What-if panel
st.subheader("Try 'What‑If' Prediction")
with st.expander("Adjust neighborhood features"):
    col1, col2, col3 = st.columns(3)
    inputs = {}
    with col1:
        inputs["MedInc"] = st.slider("Median Income (10k$)", 0.5, 15.0, 5.0, 0.1)
        inputs["HouseAge"] = st.slider("House Age (years)", 1.0, 52.0, 20.0, 1.0)
        inputs["AveRooms"] = st.slider("Avg Rooms", 1.0, 10.0, 5.0, 0.1)
    with col2:
        inputs["AveBedrms"] = st.slider("Avg Bedrooms", 0.5, 5.0, 1.0, 0.1)
        inputs["Population"] = st.slider("Population", 100.0, 5000.0, 1000.0, 10.0)
        inputs["AveOccup"] = st.slider("Avg Occupancy", 1.0, 6.0, 3.0, 0.1)
    with col3:
        inputs["Latitude"] = st.slider("Latitude", 32.0, 42.0, 34.0, 0.1)
        inputs["Longitude"] = st.slider("Longitude", -124.0, -114.0, -120.0, 0.1)

    X_in = pd.DataFrame([inputs])[feature_names]

model_choice = st.selectbox("Model for prediction", ["RandomForest", "LinearRegression"])
model = rf if model_choice == "RandomForest" else lr
pred = model.predict(X_in)[0]
st.metric("Predicted Median House Value (×100k USD)", f"{pred:0.2f}")

# Feature importance (RF only)
st.subheader("Feature Importance (Random Forest)")
if hasattr(rf, "feature_importances_"):
    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    fig = plt.figure(figsize=(6, 4))
    fi.head(8).iloc[::-1].plot(kind="barh")
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("Feature importance not available for the selected model.")

# Quick EDA
st.subheader("Quick EDA")
st.write("Sample of the dataset:")
st.dataframe(pd.concat([X, y.rename("MedHouseVal")], axis=1).head(10))
