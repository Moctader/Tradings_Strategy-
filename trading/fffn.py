import pandas as pd
from darts import TimeSeries
from darts.models import NLinearModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, rmse, smape, r2_score
import matplotlib.pyplot as plt
import ffn
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def get_all_input_and_target_timeseries(self, df, target_column):
        pass

    @abstractmethod
    def split_train_test(self, input_series: TimeSeries, target_series: TimeSeries, split_ratio=0.8):
        pass

class DefaultDataProcessor(DataProcessor):
    def get_all_input_and_target_timeseries(self, df, target_column):
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)
        input_columns = [col for col in df.columns if col != target_column]
        input_series = TimeSeries.from_dataframe(df, value_cols=input_columns, fill_missing_dates=False, freq=None)
        if target_column not in df.columns:
            raise ValueError(f"The target column '{target_column}' is missing from the DataFrame.")
        target_series = TimeSeries.from_dataframe(df, value_cols=target_column, fill_missing_dates=False, freq=None)
        return input_series, target_series

    def split_train_test(self, input_series: TimeSeries, target_series: TimeSeries, split_ratio=0.8):
        split_index = int(len(input_series) * split_ratio)
        input_train, input_test = input_series.split_before(split_index)
        target_train, target_test = target_series.split_before(split_index)
        return input_train, input_test, target_train, target_test

class SignalGenerator(ABC):
    @abstractmethod
    def generate_signals(self, y_pred, window=3, upper_threshold=0.000051, lower_threshold=-0.000051):
        pass

class MovingAverageSignalGenerator(SignalGenerator):
    def generate_signals(self, y_pred, window=3, upper_threshold=0.000051, lower_threshold=-0.000051):
        deviation = self.moving_average_deviation(y_pred.pd_series(), window)
        deviation_series = deviation
        signals = []
        for dev in deviation_series:
            if dev > upper_threshold:
                signals.append('Buy')
            elif dev < lower_threshold:
                signals.append('Sell')
            else:
                signals.append('Exit')
        times = deviation_series.index
        return signals, times, deviation_series

    def moving_average_deviation(self, data, window=3):
        data_series = data
        moving_avg = data_series.rolling(window=window).mean()
        deviation = data_series - moving_avg
        return deviation

class Strategy(ABC):
    @abstractmethod
    def apply_strategy(self, times, prices, signals):
        pass

class SimpleStrategy(Strategy):
    def apply_strategy(self, times, prices, signals):
        data = pd.DataFrame(index=times)
        data['Price'] = prices
        data['Signal'] = signals

        positions = []
        position = 0
        for i in range(len(data)):
            signal = data['Signal'].iloc[i]
            if signal == 'Buy':
                if position >= 0:
                    position += 1
                else:
                    position = 1
            elif signal == 'Sell':
                if position <= 0:
                    position -= 1
                else:
                    position = -1
            elif signal == 'Exit':
                position = 0
            positions.append(position)
        data['Position'] = positions

        data['Price_Change'] = data['Price'].pct_change()
        data['Position_shifted'] = data['Position'].shift(1)
        data['Position_shifted'].fillna(0, inplace=True)
        data['Strategy_Returns'] = data['Position_shifted'] * data['Price_Change']
        return data

class TradingModel:
    def __init__(self, model_path, data_path, target_column, test_size=10000):
        self.model_path = model_path
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.scaler_X = Scaler()
        self.scaler_y = Scaler()
        self.model = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_scaled = None
        self.y_test_scaled = None
        self.y_pred = None
        self.y_test_actual = None
        self.data_processor = DefaultDataProcessor()
        self.signal_generator = MovingAverageSignalGenerator()
        self.strategy = SimpleStrategy()

    def load_data(self):
        df = pd.read_pickle(self.data_path).head(100000)
        input_ts, target_ts = self.data_processor.get_all_input_and_target_timeseries(df, self.target_column)
        X_train, X_test, y_train, y_test = self.data_processor.split_train_test(input_ts, target_ts, split_ratio=0.8)
        self.X_test = X_test[:self.test_size]
        self.y_test = y_test[:self.test_size]
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        self.y_train_scaled = self.scaler_y.fit_transform(y_train)
        self.y_test_scaled = self.scaler_y.transform(self.y_test)

    def load_model(self):
        self.model = NLinearModel.load(self.model_path)

    def predict(self):
        y_pred_scaled = self.model.historical_forecasts(
            series=self.y_test_scaled,
            past_covariates=self.X_test_scaled,
            start=self.y_test_scaled.start_time(),
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=True,
            last_points_only=True,
        )
        self.y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        self.y_test_actual = self.scaler_y.inverse_transform(self.y_test_scaled)

    def evaluate(self):
        r2_sc = r2_score(self.y_test_actual, self.y_pred)
        print(f"R2: {r2_sc}")
        rmse_er = rmse(self.y_test_actual, self.y_pred)
        print(f"RMSE: {rmse_er}")
        mae_error = mae(self.y_test_actual, self.y_pred)
        print(f"Mean Absolute Error: {mae_error}")
        mse_error = mse(self.y_test_actual, self.y_pred)
        print(f"Mean Squared Error: {mse_error}")
        sMAPE_error = smape(self.y_test_actual, self.y_pred)
        print(f"sMAPE Error: {sMAPE_error}")

    def plot_actual_vs_predicted(self):
        plt.figure(figsize=(12, 6))
        y_test_actual_series = self.y_test_actual.pd_series()
        y_pred_series = self.y_pred.pd_series()
        y_test_actual_series.plot(label='Actual')
        y_pred_series.plot(label='Predicted')
        plt.legend()
        plt.title('Actual vs Predicted Values')
        plt.show()

    def generate_signals(self):
        return self.signal_generator.generate_signals(self.y_pred)

    def apply_trading_strategy(self, times, prices, signals):
        return self.strategy.apply_strategy(times, prices, signals)

    def plot_trading_signals(self, strategy_data, y_test_actual_series, y_pred_series):
        buy_signals = strategy_data[strategy_data['Signal'] == 'Buy']
        sell_signals = strategy_data[strategy_data['Signal'] == 'Sell']
        exit_signals = strategy_data[strategy_data['Signal'] == 'Exit']

        plt.figure(figsize=(12, 6))
        y_test_actual_series.plot(label='Actual')
        y_pred_series.plot(label='Predicted')

        plt.scatter(buy_signals.index, buy_signals['Price'], marker='^', color='green', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['Price'], marker='v', color='red', label='Sell Signal')
        plt.scatter(exit_signals.index, exit_signals['Price'], marker='o', color='blue', label='Exit Signal')

        plt.legend()
        plt.title('Trading Signals')
        plt.show()

    def calculate_performance(self, strategy_data):
        initial_capital = 10000
        strategy_data['Strategy_Returns'].fillna(0, inplace=True)
        strategy_data['Cumulative_Returns'] = (1 + strategy_data['Strategy_Returns']).cumprod()
        strategy_data['Equity'] = initial_capital * strategy_data['Cumulative_Returns']

        strategy_returns = strategy_data['Strategy_Returns']
        stats = ffn.PerformanceStats(strategy_returns)

        print(stats)
        print("Performance Metrics:")
        print(f"Total Return: {stats.total_return * 100:.2f}%")
        print(f"Annual Return: {stats.cagr * 100:.2f}%")
        print(f"Max Drawdown: {stats.max_drawdown * 100:.2f}%")
        print(f"Sharpe Ratio: {stats.daily_sharpe:.2f}")
        print(f"Sortino Ratio: {stats.daily_sortino:.2f}")
        print(f"Win Ratio: {stats.win_year_perc * 100:.2f}%")

        plt.figure(figsize=(12, 6))
        strategy_data['Equity'].plot(label='Equity Curve')
        plt.title('Equity Curve')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.show()




# Load the Trading Model
trading_model = TradingModel(
    model_path="./nlinear_model.pth",
    data_path="features.pickle",
    target_column='label'
)

# Load the data, model, predict, evaluate, and plot
trading_model.load_data()
trading_model.load_model()
trading_model.predict()
trading_model.evaluate()
trading_model.plot_actual_vs_predicted()

# Generate Trading Signals and Apply Trading Strategy
signals, times, deviations = trading_model.generate_signals()
total_length = len(times)
times = pd.date_range(start='2023-01-01', periods=total_length, freq='min')

y_pred_series = trading_model.y_pred.pd_series()
y_pred_series.index = times

y_test_actual_series = trading_model.y_test_actual.pd_series()[8:8 + total_length]
y_test_actual_series.index = times
deviations.index = times

prices = y_pred_series
strategy_data = trading_model.apply_trading_strategy(times, prices, signals)

# Plot the trading signals
trading_model.plot_trading_signals(strategy_data, y_test_actual_series, y_pred_series)

# Calculate performance metrics
trading_model.calculate_performance(strategy_data)