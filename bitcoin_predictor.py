# bitcoin_predictor.py

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from datetime import datetime, timedelta

def feature_engineering(df):
    """
    Creates new features from the price data to improve model accuracy.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Price' and a DatetimeIndex.
        
    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    df_features = df[['Price']].copy()
    
    # 1. Moving Averages - to capture trends
    df_features['MA_7'] = df_features['Price'].rolling(window=7).mean()
    df_features['MA_21'] = df_features['Price'].rolling(window=21).mean()
    
    # 2. Momentum - to capture the rate of change
    df_features['Momentum'] = df_features['Price'].pct_change(14)
    
    # 3. Volatility - to capture market stability
    df_features['Volatility'] = df_features['Price'].rolling(window=7).std()
    
    # Drop rows with NaN values created by rolling windows
    df_features.dropna(inplace=True)
    
    return df_features

def load_data(file, prediction_days):
    """
    Loads data, creates features, and prepares it for forecasting.
    
    Args:
        file: A file path or file-like object.
        prediction_days (int): The number of days to predict.
        
    Returns:
        tuple: Contains features (X), labels (y), the array for future prediction,
               and the original dataframe with a datetime index.
    """
    df = pd.read_csv(file)
    if 'Date' not in df.columns:
        raise ValueError("CSV file must contain a 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.set_index('Date', inplace=True)
    
    # --- NEW: Apply Feature Engineering ---
    df_processed = feature_engineering(df)
    
    # Create the label column (future price) and add it to the processed dataframe
    df_processed['Prediction'] = df_processed['Price'].shift(-prediction_days)
    
    # Define the features to be used by the model
    feature_columns = ['Price', 'MA_7', 'MA_21', 'Momentum', 'Volatility']
    
    # Create the feature set (X)
    X = np.array(df_processed[feature_columns])
    
    # Keep only rows that have a 'Prediction' value (drop last 'n' rows)
    X = X[:-prediction_days]
    
    # Create the label set (y)
    y = np.array(df_processed['Prediction'])[:-prediction_days].ravel()

    # Get the data for the actual future prediction (last 'n' rows of features)
    prediction_array = np.array(df_processed[feature_columns])[-prediction_days:]
    
    # Return the original dataframe for plotting purposes
    return X, y, prediction_array, df

def train_model(X, y):
    """
    Trains an SVR model using a chronological split and GridSearchCV.
    (This function remains the same, but now it trains on more features)
    """
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))
    xtrain, xtest = X[:split_index], X[split_index:]
    ytrain, ytest = y[:split_index], y[split_index:]

    param_grid = {
        'C': [1e3, 5e3, 1e4, 5e4],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 'auto', 'scale']
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=tscv, verbose=0, n_jobs=-1)
    grid_search.fit(xtrain, ytrain)

    best_model = grid_search.best_estimator_
    accuracy = best_model.score(xtest, ytest)
    
    return best_model, accuracy, grid_search.best_params_

def get_predictions(model, prediction_array):
    """
    Makes future predictions using the trained model.
    (This function remains the same)
    """
    return model.predict(prediction_array)
