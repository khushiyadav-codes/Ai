import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Page setup ---
st.set_page_config(page_title="Housing Price Predictor", layout="wide")
st.title("🏠 Simple Housing Price Predictor")
st.caption("California Housing dataset • Linear Regression • Quick Dashboard")

# --- Load data ---
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]   # target: house value in $100k units
features = X.columns.tolist()

# --- Quick EDA ---
st.subheader("📊 Quick Data Overview")
st.write("Dataset sample:")
st.dataframe(df.head(10))

st.write("Summary stats:")
st.dataframe(df.describe().T)

# Scatter plot
st.write("### 📈 Scatter Plot (feature vs target)")
feat = st.selectbox("Choose feature", features, index=0)
fig, ax = plt.subplots()
ax.scatter(df[feat], y, alpha=0.3)
ax.set_xlabel(feat)
ax.set_ylabel("Median House Value (×100k USD)")
st.pyplot(fig)

# --- Train model ---
st.subheader("🤖 Train Linear Regression Model")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

results = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2}])
st.write("Model performance on test set:")
st.dataframe(results.style.format("{:.3f}"))

# --- What-if prediction ---
st.subheader("🧪 Try a Custom Input")
inputs = {}
cols = st.columns(3)
for i, feat in enumerate(features):
    col = cols[i % 3]
    val = float(X[feat].mean())
    inputs[feat] = col.slider(feat, float(X[feat].min()), float(X[feat].max()), val)

X_in = pd.DataFrame([inputs])[features]
pred = lr.predict(X_in)[0]

st.metric("Predicted Median House Value", f"${pred*100000:,.0f}")

