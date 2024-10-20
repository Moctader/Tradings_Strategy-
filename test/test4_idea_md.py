# Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ffn

# 1. Data Retrieval
def get_historical_data(ticker):
    data = yf.download(ticker, start='2015-01-01', end='2024-01-01')
    return data['Close']

# 2. Feature Engineering
def create_features(data):
    # Create lag features
    data['Lag1'] = data.shift(1)
    data['Lag2'] = data.shift(2)
    data['Lag3'] = data.shift(3)
    
    # Create target variable (Next day closing price)
    data['Target'] = data.shift(-1)

    # Drop NaN values
    return data.dropna()

# 3. Model Training
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model performance
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model Mean Squared Error: {mse:.2f}')

    return model

# 4. Signal Generation
def generate_signals(data, model):
    # Get features for prediction
    features = data[['Lag1', 'Lag2', 'Lag3']].values
    data['Predicted_Close'] = model.predict(features)
    
    # Generate signals
    data['Signal'] = np.where(data['Predicted_Close'] > data['Close'], 1, 0)  # 1 for buy, 0 for sell/hold
    data['Position'] = data['Signal'].diff()  # Track position changes

    return data

# 5. Backtesting with ffn
def backtest_strategy(data):
    # Initialize portfolio
    portfolio = pd.Series(index=data.index)
    portfolio[data['Position'] == 1] = 1  # Buy signal
    portfolio[data['Position'] == -1] = 0  # Sell signal
    portfolio.fillna(method='ffill', inplace=True)  # Carry forward positions

    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = portfolio.shift(1) * data['Daily_Return']  # Strategy returns based on position

    # Calculate cumulative returns
    cumulative_returns = (1 + data['Strategy_Return']).cumprod() - 1
    cumulative_market_returns = (1 + data['Daily_Return']).cumprod() - 1
    
    # Create a performance report
    performance = pd.DataFrame({
        'Cumulative Market Returns': cumulative_market_returns,
        'Cumulative Strategy Returns': cumulative_returns
    })
    
    print(performance.tail())
    
    # Using ffn to analyze performance
    stats = ffn.merge(performance['Cumulative Market Returns'], performance['Cumulative Strategy Returns']).dropna()
    stats.columns = ['Market', 'Strategy']
    stats_stats = stats.calc_stats()
    print(stats_stats)

# Putting it all together
ticker = 'AAPL'  # Example stock ticker
close_prices = get_historical_data(ticker)
features_data = create_features(close_prices)
model = train_model(features_data[['Lag1', 'Lag2', 'Lag3']], features_data['Target'])
signals_data = generate_signals(features_data, model)
backtest_strategy(signals_data)
