import pandas as pd
from darts import TimeSeries
from darts.models import NLinearModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, rmse, smape, r2_score
import torch
import matplotlib.pyplot as plt
import os

# Define necessary functions
def get_all_input_and_target_timeseries(df, target_column):
  df = df.copy()
  if isinstance(df.index, pd.DatetimeIndex):
      df = df.reset_index(drop=True)
  
  # Exclude the target column from the list of input columns
  input_columns = [col for col in df.columns if col != target_column]
  
  # Create the input TimeSeries using the input columns
  input_series = TimeSeries.from_dataframe(df, value_cols=input_columns, fill_missing_dates=False, freq=None)
  
  # Create the target TimeSeries using the target column
  if target_column not in df.columns:
      raise ValueError(f"The target column '{target_column}' is missing from the DataFrame.")
  target_series = TimeSeries.from_dataframe(df, value_cols=target_column, fill_missing_dates=False, freq=None)
  
  return input_series, target_series

def split_train_test(input_series: TimeSeries, target_series: TimeSeries, split_ratio=0.8):
  # Calculate the split index
  split_index = int(len(input_series) * split_ratio)
  
  # Use split_before and split_after with the calculated index
  input_train, input_test = input_series.split_before(split_index)
  target_train, target_test = target_series.split_before(split_index)
  
  return input_train, input_test, target_train, target_test

def moving_average_deviation(data, window=3):
  """
  Calculates the deviation of the data from its moving average.

  Parameters:
  - data: TimeSeries object
  - window: Window size for the moving average

  Returns:
  - deviation: Pandas Series of deviations
  """
  # Convert TimeSeries to pandas Series
  data_series = data.pd_series()
  
  # Calculate moving average
  moving_avg = data_series.rolling(window=window).mean()
  
  # Calculate deviation
  deviation = data_series - moving_avg
  
  return deviation

# Load and prepare the data
df = pd.read_pickle("features.pickle").head(100000)

target_column = 'label'
input_ts, target_ts = get_all_input_and_target_timeseries(df, target_column)

X_train, X_test, y_train, y_test = split_train_test(input_ts, target_ts, split_ratio=0.8)

# Limit test data to first 10000 data points
X_test = X_test[:10000]
y_test = y_test[:10000]

# Initialize scalers
scaler_X = Scaler()
scaler_y = Scaler()

# Fit scalers on training data and transform both training and test data
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Load the trained model
model = NLinearModel.load("nlinear_model.pth")

# Predict over the limited test set using historical forecasts
y_pred_scaled = model.historical_forecasts(
  series=y_test_scaled,
  past_covariates=X_test_scaled,
  start=y_test_scaled.start_time(),
  forecast_horizon=1,
  stride=1,
  retrain=False,
  verbose=True,
  last_points_only=True,
)

# Inverse transform the scaled data to get actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Evaluation metrics
r2_sc = r2_score(y_test_actual, y_pred)
print(f"R2: {r2_sc}")
rmse_er = rmse(y_test_actual, y_pred)
print(f"RMSE: {rmse_er}")
mae_error = mae(y_test_actual, y_pred)
print(f"Mean Absolute Error: {mae_error}")
mse_error = mse(y_test_actual, y_pred)
print(f"Mean Squared Error: {mse_error}")
sMAPE_error = smape(y_test_actual, y_pred)
print(f"sMAPE Error: {sMAPE_error}")

# Plot the actual vs predicted values
plt.figure(figsize=(12, 6))
y_test_actual.plot(label='Actual')
y_pred.plot(label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.show()

################################################################################
# Manual Trading Decision Based on the Forecast
################################################################################

# Calculate moving average deviation on the predicted values with window size 3
deviation = moving_average_deviation(y_pred, window=3)

# Generate trading signals based on the deviation thresholds
signals = []
for dev in deviation:
  if dev > 0.000051:
      signals.append('Buy')
  elif dev < -0.000051:
      signals.append('Sell')
  else:
      signals.append('Exit')

# Extract the time indices
times = deviation.index

# Extract the times and prices for each signal
buy_times = [times[i] for i in range(len(signals)) if signals[i] == 'Buy']
sell_times = [times[i] for i in range(len(signals)) if signals[i] == 'Sell']
exit_times = [times[i] for i in range(len(signals)) if signals[i] == 'Exit']

# Corresponding prices at the signal times
y_pred_series = y_pred.pd_series()
buy_prices = y_pred_series.loc[buy_times]
sell_prices = y_pred_series.loc[sell_times]
exit_prices = y_pred_series.loc[exit_times]

# Plot the actual vs predicted values and the signals
plt.figure(figsize=(12, 6))
y_test_actual.plot(label='Actual')
y_pred.plot(label='Predicted')

plt.scatter(buy_times, buy_prices, marker='^', color='green', label='Buy Signal')
plt.scatter(sell_times, sell_prices, marker='v', color='red', label='Sell Signal')
plt.scatter(exit_times, exit_prices, marker='o', color='blue', label='Exit Signal')

plt.legend()
plt.title('Trading Signals')
plt.show()