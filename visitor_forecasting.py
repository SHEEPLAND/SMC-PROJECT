import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error



# Settings

LATITUDE = 39.4699
LONGITUDE = -0.3763
TIMEZONE = "Europe/Madrid"
CAPACITY_LIMIT = 400

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



# 1. Download weather data

def get_weather_data():
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
    weather_json = response.json()

    weather = pd.DataFrame({
        "timestamp": weather_json["hourly"]["time"],
        "temperature": weather_json["hourly"]["temperature_2m"],
        "rainfall": weather_json["hourly"]["precipitation"]
    })

    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    return weather


# 2. Create simulated visitor data

def create_visitor_data(weather):
    np.random.seed(42)

    data = weather.copy()

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek + 1
    data["is_weekend"] = data["day_of_week"].isin([6, 7]).astype(int)

    # Simulated holiday period
    data["holiday"] = 0
    data.loc[30:36, "holiday"] = 1

    # Visitor count simulation
    data["visitor_count"] = (
        80
        + data["hour"] * 7
        + data["temperature"] * 3
        - data["rainfall"] * 10
        + data["is_weekend"] * 60
        + data["holiday"] * 80
        + np.random.normal(0, 25, len(data))
    )

    data["visitor_count"] = data["visitor_count"].round()
    data["visitor_count"] = data["visitor_count"].clip(lower=0)

    return data


# 3. Prepare features

def prepare_data(data):
    data["previous_hour_visitors"] = data["visitor_count"].shift(1)

    data["rolling_mean_3h"] = data["visitor_count"].rolling(window=3).mean()

    data = data.dropna().reset_index(drop=True)

    features = [
        "previous_hour_visitors",
        "rolling_mean_3h",
        "hour",
        "day_of_week",
        "temperature",
        "rainfall",
        "holiday"
    ]

    target = "visitor_count"

    return data, features, target


# 4. Calculate MAPE

def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    non_zero_values = actual != 0

    mape = np.mean(
        np.abs((actual[non_zero_values] - predicted[non_zero_values]) / actual[non_zero_values])
    ) * 100

    return mape


# 5. Evaluate model

def evaluate_model(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = calculate_mape(actual, predicted)

    print(f"\n{model_name}")
    print("-" * 30)
    print(f"MAE:  {mae:.2f} visitors")
    print(f"RMSE: {rmse:.2f} visitors")
    print(f"MAPE: {mape:.2f}%")

    return mae, rmse, mape



# 6. Add warning level

def add_capacity_warning(results):
    results["capacity_usage_percent"] = (
        results["predicted_visitors"] / CAPACITY_LIMIT
    ) * 100

    results["warning_level"] = "GREEN"

    results.loc[
        results["capacity_usage_percent"] >= 70,
        "warning_level"
    ] = "YELLOW"

    results.loc[
        results["capacity_usage_percent"] >= 90,
        "warning_level"
    ] = "RED"

    return results



# 7. Plot actual vs predicted

def plot_actual_vs_predicted(results, mae, mape):
    plt.figure(figsize=(12, 6))

    plt.plot(
        results["timestamp"],
        results["actual_visitors"],
        label="Actual visitors"
    )

    plt.plot(
        results["timestamp"],
        results["predicted_visitors"],
        label="Predicted visitors"
    )

    plt.axhline(
        y=CAPACITY_LIMIT,
        linestyle="--",
        label="Capacity limit"
    )

    plt.xlabel("Timestamp (Hourly)")
    plt.ylabel("Number of Visitors")
    plt.title("Actual vs Predicted Visitor Volume")

    plt.text(
    results["timestamp"].iloc[int(len(results)*0.02)],
    results["actual_visitors"].max() * 0.95,
    f"MAE = {mae:.2f} visitors\nMAPE = {mape:.2f}%",
    fontsize=10
    )

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FOLDER}/actual_vs_predicted.png")
    plt.show()



# 8. Plot average visitors by hour

def plot_average_by_hour(data):
    hourly_average = data.groupby("hour")["visitor_count"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_average.index, hourly_average.values)

    plt.xlabel("Hour of Day")
    plt.ylabel("Average Number of Visitors")
    plt.title("Average Visitor Volume by Hour")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/average_visitors_by_hour.png")
    plt.show()



# Main program

def main():
    print("Step 1: Downloading weather data...")
    weather = get_weather_data()

    print("Step 2: Creating simulated visitor data...")
    data = create_visitor_data(weather)

    print("Step 3: Preparing features...")
    data, features, target = prepare_data(data)

    # Train-test split
    train_size = int(len(data) * 0.8)

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:].copy()

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    print("Step 4: Training linear regression model")
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Step 5: Making predictions")
    predicted_values = model.predict(X_test)
    predicted_values = np.clip(predicted_values, 0, None)

    results = pd.DataFrame({
        "timestamp": test_data["timestamp"],
        "actual_visitors": y_test,
        "predicted_visitors": predicted_values
    })

    results = add_capacity_warning(results)

    print("Step 6: Evaluating model")
    mae, rmse, mape = evaluate_model(
        results["actual_visitors"],
        results["predicted_visitors"],
        "Linear Regression Results"
    )

    print("Step 7: Saving files")
    data.to_csv(f"{OUTPUT_FOLDER}/full_dataset.csv", index=False)
    results.to_csv(f"{OUTPUT_FOLDER}/forecast_results.csv", index=False)

    print("Step 8: Creating graphs")
    plot_actual_vs_predicted(results, mae, mape)
    plot_average_by_hour(data)

    print("\nDone! Files saved in the outputs folder.")


if __name__ == "__main__":
    main()