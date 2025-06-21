
import streamlit as st
import pandas as pd
import predictor


st.set_page_config(page_title="📈 Bitcoin Price Predictor", layout="centered")
st.title("🪙 Bitcoin Price Predictor")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your bitcoin.csv file", type="csv")

if uploaded_file:
    # Load and process data
    X, y, prediction_array, df = predictor.load_data(uploaded_file)

    # Train the model
    model, accuracy = predictor.train_model(X, y)

    # Predict prices for the next 30 days
    predicted_prices = predictor.predict_future(model, prediction_array)

    # Show model accuracy
    st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

    # Show last known price
    latest_price = df['Price'].iloc[-31]
    st.info(f"💰 Last known Bitcoin price: **${latest_price:.2f}**")

    # Show future predictions
    st.subheader("📊 Predicted Prices for Next 30 Days")
    future_df = pd.DataFrame(predicted_prices, columns=["Predicted Price"])
    st.dataframe(future_df)

    st.line_chart(future_df)
