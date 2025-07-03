
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Google_Stock_Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# -------------------------------
# Train-Test Split
# -------------------------------
train_size = int(len(df) * 0.9)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# -------------------------------
# ARIMA Forecast
# -------------------------------
model_arima = ARIMA(train['Close'], order=(5, 1, 0))
result_arima = model_arima.fit()
forecast_arima = result_arima.forecast(steps=len(test))

rmse_arima = sqrt(mean_squared_error(test['Close'], forecast_arima))
mae_arima = mean_absolute_error(test['Close'], forecast_arima)
r2_arima = r2_score(test['Close'], forecast_arima)

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index, test['Close'], label='Actual')
plt.plot(test.index, forecast_arima, label='ARIMA Forecast')
plt.title("ARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# SARIMA Forecast
# -------------------------------
model_sarima = SARIMAX(train['Close'], order=(1,1,1), seasonal_order=(1,1,0,12))
result_sarima = model_sarima.fit()
forecast_sarima = result_sarima.forecast(steps=len(test))

rmse_sarima = sqrt(mean_squared_error(test['Close'], forecast_sarima))
mae_sarima = mean_absolute_error(test['Close'], forecast_sarima)
r2_sarima = r2_score(test['Close'], forecast_sarima)

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index, test['Close'], label='Actual')
plt.plot(test.index, forecast_sarima, label='SARIMA Forecast')
plt.title("SARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Prophet Forecast
# -------------------------------
prophet_df = train.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
model_prophet = Prophet()
model_prophet.fit(prophet_df)
future = model_prophet.make_future_dataframe(periods=len(test))
forecast_prophet = model_prophet.predict(future)

forecast_prophet_tail = forecast_prophet['yhat'].tail(len(test))

rmse_prophet = sqrt(mean_squared_error(test['Close'], forecast_prophet_tail))
mae_prophet = mean_absolute_error(test['Close'], forecast_prophet_tail)
r2_prophet = r2_score(test['Close'], forecast_prophet_tail)

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index, test['Close'], label='Actual')
plt.plot(test.index, forecast_prophet_tail, label='Prophet Forecast')
plt.title("Prophet Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# LSTM Forecast (Improved)
# -------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])
X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test = X[:train_size-60], X[train_size-60:]
y_train, y_test = y[:train_size-60], y[train_size-60:]

model_lstm = Sequential()
model_lstm.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(100))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

model_lstm.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

predicted_lstm = model_lstm.predict(X_test)
predicted_lstm = scaler.inverse_transform(predicted_lstm)
real_lstm = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse_lstm = sqrt(mean_squared_error(real_lstm, predicted_lstm))
mae_lstm = mean_absolute_error(real_lstm, predicted_lstm)
r2_lstm = r2_score(real_lstm, predicted_lstm)

plt.figure(figsize=(12, 6))
plt.plot(test.index, real_lstm, label='Actual')
plt.plot(test.index, predicted_lstm, label='LSTM Forecast')
plt.title("LSTM Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Final Combined Results
# -------------------------------
print("\n📊 Model Performance Comparison:\n")
print(f"{'Model':<10}{'RMSE':>10}{'MAE':>10}{'R²':>10}")
print("-" * 35)
print(f"{'ARIMA':<10}{rmse_arima:>10.4f}{mae_arima:>10.4f}{r2_arima:>10.4f}")
print(f"{'SARIMA':<10}{rmse_sarima:>10.4f}{mae_sarima:>10.4f}{r2_sarima:>10.4f}")
print(f"{'Prophet':<10}{rmse_prophet:>10.4f}{mae_prophet:>10.4f}{r2_prophet:>10.4f}")
print(f"{'LSTM':<10}{rmse_lstm:>10.4f}{mae_lstm:>10.4f}{r2_lstm:>10.4f}")

# -------------------------------
# 📈 Final Comparison Plot
# -------------------------------
plt.figure(figsize=(15, 10))
plt.plot(test.index, test['Close'], color='black', label='Actual')
plt.plot(test.index, forecast_arima, label='ARIMA', linestyle='--')
plt.plot(test.index, forecast_sarima, label='SARIMA', linestyle='--')
plt.plot(test.index, forecast_prophet_tail, label='Prophet', linestyle='--')
plt.plot(test.index, predicted_lstm, label='LSTM', linestyle='--')
plt.title('Stock Price Prediction: Actual vs Forecasted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
