from utils import DATA_DIR
import requests
import pandas as pd
from datetime import datetime

ELIA_IMBALANCE_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods134/records"
ELIA_FORECAST_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods161/records"
ENTSOE_DAM_URL = "https://newtransparency.entsoe.eu/api/v1/energyPrices"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_latest_imbalance_forecast():
    params = {"limit": 1, "order_by": "datetime DESC"}
    r = requests.get(ELIA_FORECAST_URL, params=params)
    r.raise_for_status()
    record = r.json()["results"][0]
    return {
        "timestamp": record["datetime"],
        "price_eur_mwh": record["imbalanceprice"]
    }
def fetch_weather(lat=50.85, lon=4.35):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,cloudcover,windspeed_10m,shortwave_radiation",
        "timezone": "UTC"
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    return df


def load_historical_data():
    dam_path = DATA_DIR / "dam_prices.csv"
    imbalance_path = DATA_DIR / "imbalance_actual.csv"
    forecast_path = DATA_DIR / "imbalance_forecast.csv"

    dam = pd.read_csv(dam_path)
    imbalance = pd.read_csv(imbalance_path)
    forecast = pd.read_csv(forecast_path)

    return dam, imbalance, forecast


