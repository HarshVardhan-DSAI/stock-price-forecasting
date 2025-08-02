#  Stock Price Forecasting using Time Series Models

Forecasting stock prices using historical data and time series modeling techniques to aid investment decisions and trend analysis.

---

## **Project Goal**

To develop a forecasting pipeline using historical stock data and time series models to:
- Understand stock trends and seasonality
- Compare model accuracy
- Provide future price projections via an interactive streamlit

---

## **Dataset**

Historical daily stock data for **Apple Inc. (AAPL)** from **2013 to 2023**, retrieved using the `yfinance` API.

| Date       | Open   | High   | Low    | Close  | Volume    |
|------------|--------|--------|--------|--------|-----------|
| 2013-01-02 | 79.38  | 79.57  | 78.18  | 78.43  | 140129500 |
| 2013-01-03 | 78.98  | 79.25  | 77.72  | 78.18  | 113432900 |
| ...        | ...    | ...    | ...    | ...    | ...       |

*(Source: Yahoo Finance via `yfinance` Python package)*

---

## **Method**

- Data acquisition with `yfinance`
- Preprocessing and checking stationarity
- Time series decomposition
- Model building using:
  - **ARIMA**
  - **SARIMA**
  - **Prophet**
- Performance comparison using RMSE and MAPE
- Built and deployed an interactive **Streamlit** app for visualization

---

## **Results**

- **SARIMA** model outperformed others:
  - **MAPE**: 4.57%
  - **RMSE**: 9.81
- The Streamlit dashboard allows:
  - Model comparison
  - Forecast exploration

---

## **How to Run**

1. Clone the repository:
```bash
git clone https://github.com/HarshVardhan-DSAI/stock-price-forecasting.git
cd stock-price-forecasting
