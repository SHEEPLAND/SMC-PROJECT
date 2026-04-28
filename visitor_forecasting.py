import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Project settings
# -----------------------------
LATITUDE = 39.4699      # Valencia
LONGITUDE = -0.3763
TIMEZONE = "Europe/Madrid"
CAPACITY_LIMIT = 400

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -----------------------------
# 1. Get weather data from Open-Meteo
# -----------------------------
def get_openmeteo_weather():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,precipitation",
        "forecast_days": 7,
        "timezone": TIMEZONE
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    weather = pd.DataFrame({
        "timestamp": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "rainfall": data["hourly"]["precipitation"]
    })

    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    return weather


# -----------------------------
# 2. Simulate visitor data
# Replace this later with real IoT sensor data if available
# -----------------------------
def create_visitor_data(weather):
    np.random.seed(42)

    data = weather.copy()

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek + 1
    data["is_weekend"] = data["day_of_week"].isin([6, 7]).astype(int)

    # Example holiday values
    data["holiday"] = 0
    data.loc[data.index[30:36], "holiday"] = 1

    # Simulated visitor count based on realistic factors
    data["visitor_count"] = (
        80
        + data["hour"] * 7
        + data["temperature"] * 3
        - data["rainfall"] * 10
        + data["is_weekend"] * 60
        + data["holiday"] * 80
        + np.random.normal(0, 25, len(data))
    )

    data["visitor_count"] = data["visitor_count"].round().clip(lower=0)
    return data


# -----------------------------
# 3. Feature engineering
# -----------------------------
def prepare_features(data):
    data["previous_hour_visitors"] = data["visitor_count"].shift(1)
    data = data.dropna().reset_index(drop=True)

    features = [
        "previous_hour_visitors",
        "hour",
        "day_of_week",
        "temperature",
        "rainfall",
        "holiday"
    ]

    target = "visitor_count"

    return data, features, target


# -----------------------------
# 4. Evaluation metric
# -----------------------------
def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.2f} visitors")
    print(f"RMSE: {rmse:.2f} visitors")
    print(f"MAPE: {mape:.2f}%")

    return mae, rmse, mape


# -----------------------------
# 5. Main forecasting process
# -----------------------------
def main():
    print("Downloading weather data from Open-Meteo...")
    weather = get_openmeteo_weather()

    print("Creating visitor dataset...")
    data = create_visitor_data(weather)

    print("Preparing features...")
    data, features, target = prepare_features(data)

    # Train/test split
    train_size = int(len(data) * 0.8)

    train = data.iloc[:train_size].copy()
    test = data.iloc[train_size:].copy()

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # Baseline model: persistence
    test["persistence_prediction"] = test["previous_hour_visitors"]

    # Main model: Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    test["linear_regression_prediction"] = model.predict(X_test)
    test["linear_regression_prediction"] = test["linear_regression_prediction"].clip(lower=0)

    # Evaluation
    evaluate_model(
        "Persistence Baseline Results",
        y_test,
        test["persistence_prediction"]
    )

    evaluate_model(
        "Linear Regression Results",
        y_test,
        test["linear_regression_prediction"]
    )

    # Capacity warning system
    test["capacity_usage_percent"] = (
        test["linear_regression_prediction"] / CAPACITY_LIMIT
    ) * 100

    test["warning_level"] = np.where(
        test["capacity_usage_percent"] >= 90,
        "RED",
        np.where(test["capacity_usage_percent"] >= 70, "YELLOW", "GREEN")
    )

    # Save results
    data.to_csv(os.path.join(OUTPUT_FOLDER, "full_dataset.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_FOLDER, "forecast_results.csv"), index=False)

    # Plot 1: actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(test["timestamp"], y_test, label="Actual visitors")
    plt.plot(test["timestamp"], test["linear_regression_prediction"], label="Predicted visitors")
    plt.axhline(y=CAPACITY_LIMIT, linestyle="--", label="Capacity limit")
    plt.xlabel("Time")
    plt.ylabel("Visitor count")
    plt.title("Actual vs Predicted Visitor Volume")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "actual_vs_predicted.png"))
    plt.show()

    # Plot 2: visitor volume by hour
    hourly_average = data.groupby("hour")["visitor_count"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_average.index, hourly_average.values)
    plt.xlabel("Hour of day")
    plt.ylabel("Average visitor count")
    plt.title("Average Visitor Volume by Hour")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "average_visitors_by_hour.png"))
    plt.show()

    print("\nFiles saved in the outputs folder:")
    print("- full_dataset.csv")
    print("- forecast_results.csv")
    print("- actual_vs_predicted.png")
    print("- average_visitors_by_hour.png")


if __name__ == "__main__":
    main()