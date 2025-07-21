import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from datetime import datetime, timedelta
import io

# ===============================
# APP TITLE AND DESCRIPTION
# ===============================
st.title("📈 Bitcoin Price Prediction App")
st.markdown("""
This app uses a Support Vector Regression (SVR) model to predict the price of Bitcoin for a selected number of future days.
- You can **adjust the number of prediction days** in the sidebar.
- You can **upload your own data** with 'Date' and 'Price' columns.
- The model uses **GridSearchCV** to find the best hyperparameters for higher accuracy.
""")

# ===============================
# REUSABLE FUNCTIONS (adapted for Streamlit)
# ===============================
@st.cache_data # Cache the data loading to speed up reruns
def load_data(file, prediction_days):
    """Loads and prepares the data for time-series forecasting."""
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.set_index('Date', inplace=True)
    
    df['Prediction'] = df[['Price']].shift(-prediction_days)
    X = np.array(df[['Price']])[:-prediction_days]
    y = np.array(df['Prediction'])[:-prediction_days]
    prediction_array = np.array(df[['Price']])[-prediction_days:]
    
    return X, y, prediction_array, df

@st.cache_resource # Cache the trained model
def train_model(X, y):
    """Trains an SVR model using GridSearchCV."""
    # Chronological split
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))
    xtrain, xtest = X[:split_index], X[split_index:]
    ytrain, ytest = y[:split_index], y[split_index:]

    param_grid = {
        'C': [1e3, 5e3, 1e4, 5e4],
        'gamma': [0.0001, 0.0005, 0.001, 0.005]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=tscv, verbose=0, n_jobs=-1)
    grid_search.fit(xtrain, ytrain)
    
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(xtest, ytest)
    
    return best_model, accuracy, grid_search.best_params_

def create_plot(original_df, forecast_df, prediction_days):
    """Creates a Matplotlib plot of historical data and forecast."""
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.style.use('fivethirtyeight')
    
    ax.plot(original_df.index, original_df['Price'], label='Historical Price', color='royalblue')
    ax.plot(forecast_df.index, forecast_df['Forecast'], label=f'{prediction_days}-Day Forecast', color='orangered', linestyle='--')
    
    ax.set_title(f'Bitcoin Price: History and {prediction_days}-Day Forecast', fontsize=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (USD)', fontsize=14)
    ax.legend()
    ax.grid(True)
    
    return fig

# ===============================
# SIDEBAR FOR USER INPUT
# ===============================
st.sidebar.header("⚙️ User Input Parameters")

# Slider for prediction days
prediction_days = st.sidebar.slider('Select number of days to predict', 7, 60, 30)

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your own CSV file (optional)",
    type=['csv']
)

# Use uploaded file or default
if uploaded_file is not None:
    data_file = uploaded_file
    st.sidebar.success("Uploaded CSV successfully!")
else:
    data_file = 'bitcoin.csv'
    st.sidebar.info("Using default `bitcoin.csv` dataset.")


# ===============================
# MAIN APP LOGIC
# ===============================
if st.button('🚀 Run Prediction'):
    try:
        # 1. Load data
        X, y, future_prediction_array, df = load_data(data_file, prediction_days)

        if len(X) > 0:
            with st.spinner('🔍 Finding best model parameters and training... This may take a moment.'):
                # 2. Train model
                best_svr_model, model_accuracy, best_params = train_model(X, y)

            st.success('✅ Model training complete!')

            # 3. Make predictions
            future_predictions = best_svr_model.predict(future_prediction_array)
            
            # --- DISPLAY RESULTS ---

            st.subheader("📊 Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("R-squared Score", f"{model_accuracy:.4f}")
            col2.metric("Best 'C' Param", f"{best_params['C']}")
            st.metric("Best 'gamma' Param", f"{best_params['gamma']}")

            # Create forecast dataframe
            last_date = df.index[-1]
            prediction_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, prediction_days + 1)])
            forecast_df = pd.DataFrame(index=prediction_dates, data={'Forecast': future_predictions})
            forecast_df.index.name = "Date"

            st.subheader(f"🔮 {prediction_days}-Day Price Forecast")
            st.dataframe(forecast_df.style.format("${:,.2f}"))

            st.subheader("📈 Forecast Visualization")
            fig = create_plot(df, forecast_df, prediction_days)
            st.pyplot(fig)

        else:
            st.error(f"Error: The dataset is too small to create a training set with a {prediction_days}-day prediction window. Please use a larger dataset or a smaller prediction window.")
    
    except FileNotFoundError:
        st.error("Error: `bitcoin.csv` not found. Please make sure the file is in the same directory as `app.py` or upload a file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
