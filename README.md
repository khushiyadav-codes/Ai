# AI Housing Price Predictor

Simple, educational ML app that trains models on the **California Housing** dataset (from `scikit-learn`) and
lets users interactively explore predictions and feature importance via **Streamlit**.

## Quick Start

```bash
# 1) Create & activate a virtual env (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

## What it does
- Loads the California Housing dataset (no external files needed).
- Trains **Linear Regression** and **Random Forest** models.
- Reports **MAE**, **RMSE**, **R^2** on a test split.
- Provides sliders to simulate neighborhoods and see predicted median home value.
- Displays feature importance (for the Random Forest) and a quick EDA peek.

## Repo contents
- `app.py` – Streamlit UI
- `train.py` – CLI training script that saves a model to `artifacts/model.pkl`
- `notebooks/Exploration.ipynb` – optional EDA notebook
- `requirements.txt` – dependencies
- `LICENSE` – MIT
- `.gitignore` – sensible defaults

## Deploy (Streamlit Cloud)
1. Push this folder to a new GitHub repo.
2. In Streamlit Community Cloud, create a new app pointing to `app.py` on `main`.
3. Set Python version to 3.11+ (or latest) and it will auto-install from `requirements.txt`.

## Example resume line
*Built a Streamlit-based ML web app that predicts California home prices using Linear Regression and Random Forest; implemented feature engineering, evaluation (MAE/RMSE/R^2), and interactive what‑if tooling.*
