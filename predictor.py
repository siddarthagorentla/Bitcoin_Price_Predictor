import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# ===============================
# LOAD & PREPARE DATA
# ===============================
def load_data(filename='bitcoin.csv', prediction_days=30):
    df = pd.read_csv(filename)

    # Drop the Date column if it exists
    if 'Date' in df.columns:
        df.drop(['Date'], axis=1, inplace=True)

    # Shift 'Price' column to create labels for future prediction
    df['Prediction'] = df[['Price']].shift(-prediction_days)

    # Drop the last 'prediction_days' rows for features
    X = np.array(df.drop(['Prediction'], axis=1))[:-prediction_days]

    # Drop the last 'prediction_days' rows for labels
    y = np.array(df['Prediction'])[:-prediction_days]

    # Data used for predicting future prices (last 30 rows)
    prediction_array = np.array(df.drop(['Prediction'], axis=1))[-prediction_days:]

    return X, y, prediction_array, df

# ===============================
# TRAIN MODEL
# ===============================
def train_model(X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

    model = SVR(kernel='rbf', C=1e3, gamma=0.00001)
    model.fit(xtrain, ytrain)

    accuracy = model.score(xtest, ytest)
    return model, accuracy

# ===============================
# PREDICT FUTURE PRICES
# ===============================
def predict_future(model, prediction_array):
    predictions = model.predict(prediction_array)
    return predictions