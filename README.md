# Visitor Forecasting System

This project simulates and forecasts visitor volume using weather data and a machine learning model. It predicts hourly visitor counts, evaluates performance, and generates capacity warnings.

---

## Features

- Fetches real-time weather data (temperature & rainfall)
- Simulates visitor patterns based on multiple factors
- Trains a Linear Regression model for forecasting
- Evaluates predictions using MAE, RMSE, and MAPE
- Generates capacity warning levels (Green / Yellow / Red)
- Produces visualizations for insights

---


## How It Works

### 1. Weather Data Collection
- Uses Open-Meteo API to retrieve hourly:
  - Temperature
  - Precipitation

### 2. Data Simulation
Visitor counts are generated using:
- Time of day
- Weather conditions
- Weekend effect
- Holiday boost
- Random noise

### 3. Feature Engineering
Features used:
- Previous hour visitors
- Rolling 3-hour average
- Hour of day
- Day of week
- Temperature
- Rainfall
- Holiday indicator

### 4. Model Training
- Model: Linear Regression
- Train/Test split: 80/20

### 5. Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

Example results: MAE (28 visitors) and MAPE = 11%

## Visualizations

- **Actual vs Predicted Visitors**
  - Compares real vs predicted values
  - Includes capacity limit

- **Average Visitors by Hour**
  - Shows daily visitor patterns

---

## Output Files

- `full_dataset.csv` → Complete dataset  
- `forecast_results.csv` → Predictions + warnings  
- `actual_vs_predicted.png` → Forecast comparison  
- `average_visitors_by_hour.png` → Hourly trends  

---

