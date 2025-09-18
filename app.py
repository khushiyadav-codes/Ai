import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- Page setup ----------------
st.set_page_config(page_title="Easy Housing Dashboard", layout="wide")
st.title("🏠 Easy Housing Price Dashboard")
st.caption("California Housing • quick EDA • simple ML • interactive prediction")

# ---------------- Load data ----------------
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()                # has MedInc, HouseAge, AveRooms, ..., MedHouseVal
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]                 # target: median house value in $100k units
feature_names = X.columns.tolist()

# ---------------- Quick EDA ----------------
st.subheader("📊 Quick EDA")
c1, c2 = st.columns(2)
with c1:
    st.write("Sample rows")
    st.dataframe(df.head(12))
with c2:
    st.write("Summary stats")
    st.dataframe(df.describe().T)

st.write("### 🔗 Correlation heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
st.pyplot(fig)

st.write("### 📈 Scatter (feature vs target)")
feat = st.selectbox("Feature", feature_names, index=0)
sampled = df.sample(min(2000, len(df)), random_state=42)
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.scatter(sampled[feat], sampled["MedHouseVal"], alpha=0.3)
ax2.set_xlabel(feat); ax2.set_ylabel("MedHouseVal (×100k USD)")
st.pyplot(fig2)

# ---------------- Train simple model ----------------
st.subheader("🤖 Train & Evaluate (Linear Regression)")
test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
rand = st.number_input("Random state", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=rand
)

lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))   # sklearn >=1.7: no 'squared' kw
r2   = r2_score(y_test, y_pred)

metrics_df = pd.DataFrame([{"Model": "Linear Regression", "MAE": mae, "RMSE": rmse, "R2": r2}])
st.dataframe(metrics_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}))

# ---------------- What-If Prediction ----------------
st.subheader("🧪 What-If Prediction")
avg = X.mean()
c1, c2, c3 = st.columns(3)
with c1:
    MedInc   = st.slider("Median Income (10k$)", float(X["MedInc"].min()),   float(X["MedInc"].max()),   float(avg["MedInc"]))
    HouseAge = st.slider("House Age (yrs)",      float(X["HouseAge"].min()), float(X["HouseAge"].max()), float(avg["HouseAge"]))
    AveRooms = st.slider("Avg Rooms",            float(X["AveRooms"].min()), float(X["AveRooms"].max()), float(avg["AveRooms"]))
with c2:
    AveBedrms = st.slider("Avg Bedrooms",        float(X["AveBedrms"].min()), float(X["AveBedrms"].max()), float(avg["AveBedrms"]))
    Population= st.slider("Population",          float(X["Population"].min()),float(X["Population"].max()),float(avg["Population"]))
    AveOccup  = st.slider("Avg Occupancy",       float(X["AveOccup"].min()),  float(X["AveOccup"].max()),  float(avg["AveOccup"]))
with c3:
    Latitude  = st.slider("Latitude",            float(X["Latitude"].min()),  float(X["Latitude"].max()),  float(avg["Latitude"]))
    Longitude = st.slider("Longitude",           float(X["Longitude"].min()), float(X["Longitude"].max()), float(avg["Longitude"]))

row = {
    "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms, "AveBedrms": AveBedrms,
    "Population": Population, "AveOccup": AveOccup, "Latitude": Latitude, "Longitude": Longitude
}
X_in = pd.DataFrame([row])[feature_names]
pred = lr.predict(X_in)[0]  # ×100k USD
st.metric("Predicted Median House Value", f"${pred*100000:,.0f}")
st.caption("Prediction uses a simple Linear Regression trained above.")
