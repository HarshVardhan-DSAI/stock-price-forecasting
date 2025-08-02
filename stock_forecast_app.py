import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Set page config
st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("📈 Stock Price Forecasting (ARIMA, SARIMA, Prophet)")

# --- Load Data ---
df = yf.download('AAPL', start='2013-01-01', end='2023-12-31')
df = df[['Close']].dropna()
df = df.asfreq('B')  # Business days
df = df.fillna(method='ffill')

# --- Train/Test Split ---
test_size = 60
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

# --- ARIMA ---
model_arima = ARIMA(train['Close'], order=(1, 1, 1))
res_arima = model_arima.fit()
pred_arima = res_arima.forecast(steps=test_size)
pred_arima.index = test.index

rmse_arima = np.sqrt(mean_squared_error(test['Close'], pred_arima))
mape_arima = mean_absolute_percentage_error(test['Close'], pred_arima)

# --- SARIMA ---
model_sarima = SARIMAX(train['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res_sarima = model_sarima.fit()
pred_sarima = res_sarima.forecast(steps=test_size)
pred_sarima.index = test.index

rmse_sarima = np.sqrt(mean_squared_error(test['Close'], pred_sarima))
mape_sarima = mean_absolute_percentage_error(test['Close'], pred_sarima)

# --- Prophet ---
df_prophet = df.reset_index()[['Date', 'Close']]
df_prophet.columns = ['ds', 'y']
train_prophet = df_prophet.iloc[:-test_size]
test_prophet = df_prophet.iloc[-test_size:]

model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(train_prophet)

future = model_prophet.make_future_dataframe(periods=test_size, freq='B')
forecast_prophet = model_prophet.predict(future)
forecast_prophet = forecast_prophet.set_index('ds')

# Align index safely
available_dates = forecast_prophet.index.intersection(test_prophet['ds'])
pred_prophet = forecast_prophet.loc[available_dates]['yhat'].values
actual_prophet = test_prophet[test_prophet['ds'].isin(available_dates)]['y'].values

rmse_prophet = np.sqrt(mean_squared_error(actual_prophet, pred_prophet))
mape_prophet = mean_absolute_percentage_error(actual_prophet, pred_prophet)

# --- Display Metrics ---
st.header("📊 Backtest Results (Last 60 Days)")
col1, col2, col3 = st.columns(3)
col1.metric("ARIMA RMSE", f"{rmse_arima:.2f}")
col1.metric("ARIMA MAPE", f"{mape_arima:.4f}")
col2.metric("SARIMA RMSE", f"{rmse_sarima:.2f}")
col2.metric("SARIMA MAPE", f"{mape_sarima:.4f}")
col3.metric("Prophet RMSE", f"{rmse_prophet:.2f}")
col3.metric("Prophet MAPE", f"{mape_prophet:.4f}")

# --- Plot Actual vs Predicted ---
st.subheader("📉 Actual vs Predicted (Backtest)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(test.index, test['Close'], label="Actual", color='black')
ax.plot(test.index, pred_arima, label="ARIMA", linestyle='--')
ax.plot(test.index, pred_sarima, label="SARIMA", linestyle='--')
ax.plot(available_dates, pred_prophet, label="Prophet", linestyle='--')
ax.set_title("Last 60 Days: Actual vs Predicted")
ax.legend()
st.pyplot(fig)

# --- Forecast Next 60 Days ---
st.header("🔮 Forecast Next 60 Days")

# Forecast ARIMA
forecast_arima = res_arima.forecast(steps=60)
future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=60, freq='B')
forecast_arima.index = future_index

# Forecast SARIMA
forecast_sarima = res_sarima.forecast(steps=60)
forecast_sarima.index = future_index

# Forecast Prophet (safe alignment)
future_full = model_prophet.make_future_dataframe(periods=60, freq='B')
forecast_future_prophet = model_prophet.predict(future_full)
forecast_prophet_full = forecast_future_prophet.set_index('ds')
available_future_dates = forecast_prophet_full.index.intersection(future_index)
forecast_prophet_60 = forecast_prophet_full.loc[available_future_dates]['yhat']

# Plot future forecasts
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df.index[-200:], df['Close'].iloc[-200:], label="Historical", color='gray')
ax2.plot(forecast_arima.index, forecast_arima, label="ARIMA Forecast", linestyle='--')
ax2.plot(forecast_sarima.index, forecast_sarima, label="SARIMA Forecast", linestyle='--')
ax2.plot(available_future_dates, forecast_prophet_60, label="Prophet Forecast", linestyle='--')
ax2.set_title("Next 60 Days Forecast")
ax2.legend()
st.pyplot(fig2)
