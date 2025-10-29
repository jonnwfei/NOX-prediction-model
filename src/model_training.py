import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Example model choice
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib # To save the trained model
from utils import DATA_DIR, MODELS_DIR # Assumed utilities for paths

# --- 1. SETUP: ASSUMING DATA IS ALREADY LOADED AND MERGED ---

def create_features(df):
    """
    Placeholder for the feature creation function that was called earlier.
    In a real scenario, this function should be reproducible for new data.
    (This is included just for the skeleton to be runnable, though you stated it's done).
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    return df

def train_forecasting_model(df: pd.DataFrame, target_column: str, model_name: str = 'random_forest_forecaster'):
    """
    Skeleton for the entire model training process.
    """
    print(f"Starting training for target: {target_column}...")

    # --- 2. DEFINE FEATURES (X) AND TARGET (Y) ---
    
    # Assuming 'datetime' or similar is the index/time column
    df = df.set_index('datetime').sort_index()

    # Identify features (X) and target (y)
    # Exclude the target, any other price columns you wouldn't know in advance,
    # and time-related columns that have been converted to features.
    
    # NOTE: You will need to carefully define your feature set based on your merged data.
    # This is a sample list of features derived from time/weather columns.
    
    features = [
        'hour', 'dayofweek', 'month', 'year',
        # Add your actual weather features here, e.g.:
        # 'temperature_2m', 'cloudcover', 'windspeed_10m', 'shortwave_radiation',
        # And any lag features you created
    ]
    
    X = df[features].copy()
    y = df[target_column].copy()
    
    print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    print(f"Selected Features: {features}")

    # --- 3. TIME-SERIES TRAIN/TEST SPLIT (No Random Shuffling) ---
    
    # For time-series, we split based on time, using the most recent data for testing.
    # Example: 80% for training, 20% for testing
    train_size = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Training data range: {X_train.index.min()} to {X_train.index.max()}")
    print(f"Test data range: {X_test.index.min()} to {X_test.index.max()}")

    # --- 4. MODEL SELECTION AND TRAINING ---
    
    # Initialize the chosen model
    # Consider more advanced models for time series like XGBoost, LightGBM, or even deep learning models (LSTMs).
    model = RandomForestRegressor(
        n_estimators=100,      # Number of trees in the forest
        max_depth=10,          # Max depth of the trees
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all processors
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. EVALUATION ---
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Performance on Test Set ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    # Optional: Plotting results (requires matplotlib, not included here)
    
    # --- 6. MODEL SAVING (Persistence) ---
    
    model_filepath = MODELS_DIR / f"{model_name}_{target_column}.joblib"
    joblib.dump(model, model_filepath)
    print(f"\nModel saved successfully to: {model_filepath}")
    
    return model, mse, mae


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    from data_fetcher import load_historical_data, fetch_weather
    
    # 1. Load Data
    dam, imbalance, forecast = load_historical_data()
    weather = fetch_weather() # Assuming this fetches historical weather for simplicity
    
    # 2. Pre-merge Setup (Clean up column names and ensure datetime format)
    dam.rename(columns={'Price': 'DAM_Price', 'Datetime': 'datetime'}, inplace=True)
    imbalance.rename(columns={'ImbalancePrice': 'Imbalance_Actual', 'Datetime': 'datetime'}, inplace=True)
    forecast.rename(columns={'ImbalancePrice': 'Imbalance_Forecast', 'Datetime': 'datetime'}, inplace=True)
    
    dam['datetime'] = pd.to_datetime(dam['datetime'])
    imbalance['datetime'] = pd.to_datetime(imbalance['datetime'])
    forecast['datetime'] = pd.to_datetime(forecast['datetime'])
    weather.rename(columns={'time': 'datetime'}, inplace=True)
    
    # 3. Merge DataFrames
    # Use 'outer' merge initially to ensure no data is lost, then fill NaNs or drop.
    df = pd.merge(dam, imbalance, on='datetime', how='outer')
    df = pd.merge(df, forecast, on='datetime', how='outer')
    df = pd.merge(df, weather, on='datetime', how='outer')
    
    # Assuming we want to predict Imbalance_Actual, we drop rows where it's NaN
    # and fill NaNs for weather data (e.g., with 0 or mean/median)
    target_column = 'Imbalance_Actual'
    df = df.dropna(subset=[target_column]).sort_values(by='datetime').reset_index(drop=True)
    df = df.fillna(0) # Simple imputation, improve this in real project!
    
    # 4. Feature Engineering
    df = create_features(df)
    
    # 5. Execute Training
    # Ensure MODELS_DIR exists before running
    MODELS_DIR.mkdir(exist_ok=True) 
    
    # Example: Train a model to predict 'Imbalance_Actual'
    trained_model, mse, mae = train_forecasting_model(df.copy(), target_column=target_column)