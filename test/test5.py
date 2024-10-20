# Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ffn

# Step 1: Data Retrieval
def get_historical_data(ticker):
    data = yf.download(ticker, start='2015-01-01', end='2024-01-01')
    return data['Close']

# Step 2: Feature Engineering
def create_features(data):
    # Create lag features
    for lag in range(1, 4):
        data[f'Lag{lag}'] = data.shift(lag)
    
    # Create target variable for regression (next day's closing price)
    data['Target_Close'] = data.shift(-1)
    
    # Create target variable for classification (buy/sell/hold)
    data['Price_Change'] = data['Target_Close'] - data
    data['Signal'] = np.where(data['Price_Change'] > 0, 1, 0)  # 1 for Buy, 0 for Hold/Sell
    data['Signal'] = data['Signal'].shift(1)  # Shift to avoid lookahead bias

    # Drop NaN values
    return data.dropna()

# Step 3: Model Training
def train_regression_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model performance
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model Mean Squared Error (Regression): {mse:.2f}')

    return model

def train_classification_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model performance
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy (Classification): {accuracy:.2f}')

    return model

# Step 4: Signal Generation
def generate_signals(data, regression_model, classification_model):
    # Prepare features for prediction
    feature_columns = [f'Lag{lag}' for lag in range(1, 4)]
    features = data[feature_columns].values
    
    # Regression prediction for next closing price
    data['Predicted_Close'] = regression_model.predict(features)
    
    # Classification prediction for buy/sell/hold signals
    data['Predicted_Signal'] = classification_model.predict(features)

    # Calculate position changes based on predicted signals
    data['Position'] = data['Predicted_Signal'].diff()

    return data

# Step 5: Backtesting with ffn
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

# Split data for regression and classification
features_reg = features_data[['Lag1', 'Lag2', 'Lag3']]
target_reg = features_data['Target_Close']
target_class = features_data['Signal']

# Train models
regression_model = train_regression_model(features_reg, target_reg)
classification_model = train_classification_model(features_reg, target_class)

# Generate signals and backtest the strategy
signals_data = generate_signals(features_data, regression_model, classification_model)
backtest_strategy(signals_data)
