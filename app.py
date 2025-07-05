# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Import the functions from our logic file
import bitcoin_predictor as bp

# Helper function for plotting
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
# STREAMLIT APP LAYOUT
# ===============================

st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")

st.title("📈 Bitcoin Price Prediction App")
st.markdown("""
This app uses a Support Vector Regression (SVR) model to predict the price of Bitcoin. 
The core logic is separated from the UI for better code organization.
""")

# --- Sidebar for User Input ---
st.sidebar.header("⚙️ User Input Parameters")

prediction_days = st.sidebar.slider('Select number of days to predict', 7, 60, 30)

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

# --- Main App Logic ---
if st.button('🚀 Run Prediction'):
    try:
        # 1. LOAD DATA (using the function from bitcoin_predictor.py)
        # Use a caching decorator to avoid reloading/recomputing on every interaction
        @st.cache_data
        def cached_load_data(file, days):
            return bp.load_data(file, days)
            
        X, y, future_prediction_array, df = cached_load_data(data_file, prediction_days)

        if len(X) > 0:
            with st.spinner('🔍 Finding best model parameters and training... This may take a moment.'):
                # 2. TRAIN MODEL (using the function from bitcoin_predictor.py)
                # Cache the model so it doesn't retrain if only the plot changes
                @st.cache_resource
                def cached_train_model(X_train, y_train):
                    return bp.train_model(X_train, y_train)
                
                best_svr_model, model_accuracy, best_params = cached_train_model(X, y)

            st.success('✅ Model training complete!')

            # 3. GET PREDICTIONS (using the function from bitcoin_predictor.py)
            future_predictions = bp.get_predictions(best_svr_model, future_prediction_array)
            
            # --- DISPLAY RESULTS ---
            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("R-squared Score", f"{model_accuracy:.4f}")
            col2.metric("Best 'C' Param", f"{best_params['C']}")
            col3.metric("Best 'gamma' Param", f"{best_params['gamma']}")

            # Create forecast dataframe for display
            last_date = df.index[-1]
            prediction_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, prediction_days + 1)])
            forecast_df = pd.DataFrame(index=prediction_dates, data={'Forecast': future_predictions})
            forecast_df.index.name = "Date"

            col_left, col_right = st.columns([1, 2])
            with col_left:
                st.subheader(f"🔮 {prediction_days}-Day Price Forecast")
                st.dataframe(forecast_df.style.format("${:,.2f}"))

            with col_right:
                st.subheader("📈 Forecast Visualization")
                fig = create_plot(df, forecast_df, prediction_days)
                st.pyplot(fig)

        else:
            st.error(f"Error: Dataset is too small for a {prediction_days}-day prediction window. Use a larger dataset or a smaller window.")
    
    except FileNotFoundError:
        st.error("Error: `bitcoin.csv` not found. Please place it in the same directory as `app.py` or upload a file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
