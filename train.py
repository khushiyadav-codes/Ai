import os, joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():
    os.makedirs("artifacts", exist_ok=True)
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "features": X.columns.tolist()}, "artifacts/model.pkl")
    print("Saved model to artifacts/model.pkl")

if __name__ == "__main__":
    main()
