Stock Price Forecasting Using Time Series Analysis
Project Goal
To forecast future stock prices using historical data and advanced time series models, helping investors and analysts make informed decisions based on data-driven predictions.

Dataset
We use historical stock price data (e.g., Apple Inc. - AAPL), downloaded directly from Yahoo Finance. The dataset includes daily closing prices over multiple years.

Date	Close Price (USD)
2013-01-02	16.65
2013-01-03	16.44
2013-01-04	15.98
2013-01-07	15.89
2013-01-08	15.93
...	...
Method
Downloaded historical stock prices using yfinance.

Performed exploratory data analysis and visualizations.

Conducted stationarity tests (ADF and KPSS) to understand data properties.

Decomposed the series to analyze trend and seasonality.

Applied statistical models:

ARIMA for univariate forecasting

SARIMA to incorporate seasonal components

Applied ML-based forecasting with Prophet to capture complex seasonal patterns.

Evaluated models using metrics like RMSE and MAPE.

Plotted AutoCorrelation and Partial AutoCorrelation functions to select model parameters.

Results
Successfully identified patterns and seasonality in stock price data.

Generated accurate forecasts using multiple models, visually comparing predicted vs actual values.

Provided metrics demonstrating model accuracy and reliability.

How to Run
Clone the repository:

bash
git clone https://github.com/your-username/stock-price-forecasting.git
Install required Python packages:

bash
pip install -r requirements.txt
Open the notebook:

bash
jupyter notebook Stock-price-forecasting.ipynb
Run each cell to reproduce the full analysis. Change the stock ticker as desired.

Libraries Used
yfinance for financial data download

pandas, numpy for data handling

matplotlib, seaborn for visualization

statsmodels for statistical tests and ARIMA/SARIMA models

prophet for ML time series forecasting

scikit-learn for evaluation metrics

