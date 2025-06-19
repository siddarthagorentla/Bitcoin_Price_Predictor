bitcoin_predictor/
├── bitcoin.csv
├── model/
│   └── predictor.py       ✅ (Fixed version)
├── ui/
│   └── app.py             ✅ (This file)
├── requirements.txt

Technologies Used: Python, scikit-learn, Support Vector Regression (SVR), RBF Kernel

✅ Project Summary:
A machine learning model to predict Bitcoin prices for the next 30 days using historical price data.

🎯 What to Explain in the Interview:
Dataset: Collected historical Bitcoin price data (likely from APIs like Yahoo Finance or Kaggle datasets).

Preprocessing: Cleaned missing data, normalized it.

Model: SVR with RBF kernel chosen for its ability to handle non-linear data.

Why SVR: Better than linear models for capturing market volatility.

Prediction: Built logic to forecast the next 30 days and plotted results using Matplotlib.

Evaluation: Discuss Mean Squared Error or R² score to evaluate accuracy.
