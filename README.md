# Wind-Power-Forecasting
# 🌬️ Wind Turbine Power Forecasting Dashboard

An interactive ML dashboard for wind turbine power forecasting. Upload a turbine SCADA/weather CSV, automatically train and compare 7 machine learning models, and explore performance through an interactive Streamlit interface.


---

## Features

- **Automated ML Pipeline** — preprocessing, feature engineering, and training 7 models (Linear Regression, Lasso, Ridge, Random Forest, XGBoost, LightGBM, CatBoost) in one upload
- **Data Quality Report** — missing values, duplicates, and outlier detection/capping
- **7 Dashboard Pages** — energy production trends, wind & climate analysis, power curve, model accuracy, and real-time forecasting
- **Real-time Prediction** — adjust weather sliders and get instant kW output using the best model or an ensemble average

---

## Getting Started

```bash
git clone https://github.com/linda-mkaouar/Wind-Power-Forecasting.git
cd Wind-Power-Forecasting
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser and upload your CSV from the sidebar.

---

## Dataset Format

The CSV must contain these columns:

| Column | Description |
|---|---|
| `Time` | Timestamp (e.g. `2022-01-01 00:00`) |
| `Power` | Normalized power output — **must be 0 to 1** (divide raw kW by rated capacity) |
| `windspeed_10m` | Wind speed at 10 m (m/s) |
| `windspeed_100m` | Wind speed at 100 m (m/s) |
| `windgusts_10m` | Wind gusts at 10 m (m/s) |
| `winddirection_100m` | Wind direction at 100 m (0–360°) |
| `temperature_2m` | Temperature at 2 m (°F) |
| `relativehumidity_2m` | Relative humidity at 2 m (%) |
| `dewpoint_2m` | Dew point at 2 m (°F) |

---

## Tech Stack

- **Frontend:** Streamlit, Plotly
- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Data:** pandas, NumPy, SciPy
Dataset source: https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting
