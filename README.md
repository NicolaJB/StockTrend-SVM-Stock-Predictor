# StockTrendSVM: Stock Predictor Enhanced with Transformer Sentiment

## Overview
This project implements a classical **Support Vector Machine (SVM)** model to predict stock price direction, augmented with sentiment features derived from financial news headlines using a **transformer-based model**.  

The notebook integrates traditional technical indicators and macroeconomic data with modern NLP sentiment analysis to improve stock price movement prediction. The model is trained and tested on historical **Apple (AAPL)** stock data and is fully compatible with **Google Colab**.

**Result:** This demo achieved ~45% directional prediction accuracy on Apple (AAPL) stock data using combined technical, macroeconomic, and sentiment features.

The pipeline demonstrates an end-to-end workflow combining feature engineering, sentiment analysis, model training, and evaluation in Colab.

## Skills & Tools
Python, SVM, Transformers, NLP, Financial Data Analysis, Time Series Analysis, Scikit-learn, yfinance, FRED API, Pandas, NumPy, Matplotlib

## Features

| Capability | Description |
|-----------|-------------|
| Technical Indicators | Moving averages, volatility, returns. |
| Macroeconomic Feature | 10-Year Treasury Rate from FRED API. |
| Transformer Sentiment Analysis | Sentiment scoring of financial news headlines. |
| Classical SVM Classifier | Predicts next-day stock price direction. |
| Visualisation | Step chart of actual vs predicted price movement. |

## Repository Structure
```bash
stocktrendsvm/
├─ README.md # This file
├─ StockTrendSVM.ipynb # End-to-end notebook
├─ data/ # Optional historical stock or news data
└─ models/ # Saved SVM and feature preprocessing models
```
## Quick Start

### 1. Install Required Libraries

```python
!pip install yfinance fredapi nltk transformers datasets scikit-learn matplotlib pandas numpy --quiet
```
###  2. Import Libraries
```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from fredapi import Fred
import nltk
nltk.download('vader_lexicon')

from transformers import pipeline
```
### 3. Configure API Keys and Download Data
```python
# FRED API key
FRED_API_KEY = 'your_fred_api_key_here'  # Replace with your own key
fred = Fred(api_key=FRED_API_KEY)

# Download Apple stock data
ticker = 'AAPL'
stock_data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
```
### 4. Feature Engineering
- Compute moving averages (MA10, MA50)
- Daily returns and volatility
- 10-Year Treasury Rate
- Transformer sentiment scores for financial news headlines
```python
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
```
### 5. Prepare Target Variable
```python
# Next-day price movement
stock_data['Target'] = (stock_data['Return'].shift(-1) > 0).astype(int)
stock_data = stock_data.dropna()
```
### 6. Train SVM Classifier
```python
features = ['MA10', 'MA50', 'Volatility', 'Volume', 'InterestRate', 'SentimentScore']
X = stock_data[features]
y = stock_data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

clf = SVC(kernel='rbf', C=1.0, gamma='auto')
clf.fit(X_train, y_train)
```
### 7. Evaluate Model
```python
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
```
### 8. Visualise Predictions
```python
plt.figure(figsize=(12, 6))
plt.step(stock_data['Date'].iloc[-len(y_test):], y_test.values, label='Actual', where='mid')
plt.step(stock_data['Date'].iloc[-len(y_test):], y_pred, label='Predicted', where='mid', linestyle='--')
plt.title(f'{ticker} Price Direction Prediction with Transformer-Based Sentiment')
plt.ylabel('Direction (0 = Down, 1 = Up)')
plt.xlabel('Date')
plt.legend()
plt.show()
```
## Requirements
- Python 3.7+
- yfinance, fredapi, nltk, transformers, datasets
- scikit-learn, matplotlib, pandas, numpy
  
## Usage
- Clone or download the repository.
- Replace FRED_API_KEY with your own key.
- Run the notebook in Google Colab or any Python environment.

The notebook downloads historical stock data, fetches macroeconomic indicators, computes sentiment scores, trains an SVM model, and evaluates predictions.

## Notes
- Sentiment analysis uses a pre-trained DistilBERT model fine-tuned on SST-2.
- The sample news dataset is limited; for improved performance, use a larger, real dataset.
- [FRED API](https://fred.stlouisfed.org/) key is required to fetch macroeconomic data; without it, a fallback value is used.

## Educational Goals
This project demonstrates:
- Combining classical ML models (SVM) with modern NLP sentiment features
- Integration of technical, macroeconomic, and sentiment indicators for predictive modelling
- Reproducible, end-to-end ML pipelines suitable for financial data analysis

## Licence
This project is licensed under the Apache License 2.0. See the LICENSE file for details.
