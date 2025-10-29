from utils import DATA_DIR, MODELS_DIR
import pandas as pd
from data_fetcher import load_historical_data, fetch_weather
from datetime import datetime
# Load your datasets
dam_path = DATA_DIR / "dam_prices.csv"
imbalance_path = DATA_DIR / "imbalance_actual.csv"
forecast_path = DATA_DIR / "imbalance_forecast.csv"
dam = pd.read_csv(dam_path)
imbalance = pd.read_csv(imbalance_path)
forecast = pd.read_csv(forecast_path)

# Just show the first few rows of each dataset
print("DAM dataset:")
print(dam.head(), "\n")

print("Imbalance dataset:")
print(imbalance.head(), "\n")

print("Forecast dataset:")
print(forecast.head(), "\n")

if __name__ == "__main__":
    dam, imbalance, forecast = load_historical_data()
    weather = fetch_weather()
    df = pd.merge(dam, imbalance, forecast)
    df = pd.create_features(df)
    print(df.head())
