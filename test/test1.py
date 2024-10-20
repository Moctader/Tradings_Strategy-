import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download historical data
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Returns'] = data['Adj Close'].pct_change()
    return data

# Define breakout strategy
def generate_signals(data, window=20, breakout_threshold=2):
    data['MA'] = data['Adj Close'].rolling(window=window).mean()
    data['Volatility'] = data['Adj Close'].rolling(window=window).std()
    
    # Signal conditions
    data['Buy_Signal'] = np.where(data['Adj Close'] > data['MA'] + breakout_threshold * data['Volatility'], 1, 0)
    data['Sell_Signal'] = np.where(data['Adj Close'] < data['MA'] - breakout_threshold * data['Volatility'], -1, 0)
    
    # Combine Buy and Sell signals into a single column
    data['Signal'] = data['Buy_Signal'] + data['Sell_Signal']
    
    return data

# Backtest the strategy
def backtest_strategy(data, initial_capital=10000):
    # Initializing variables
    data['Position'] = data['Signal'].shift()  # Delay the signal by 1 to avoid look-ahead bias
    data['Market_Returns'] = data['Returns']
    
    # Calculate strategy returns based on position (long=1, short=-1, no position=0)
    data['Strategy_Returns'] = data['Position'] * data['Market_Returns']
    
    # Calculate cumulative returns for the strategy and market
    data['Cumulative_Market_Returns'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()

    # Calculate final portfolio value
    final_value = initial_capital * data['Cumulative_Strategy_Returns'][-1]
    
    return data, final_value

# Plot performance
def plot_performance(data):
    plt.figure(figsize=(14, 7))
    
    plt.plot(data['Cumulative_Market_Returns'], label='Market Returns', color='blue')
    plt.plot(data['Cumulative_Strategy_Returns'], label='Strategy Returns', color='orange')
    
    plt.title('Market vs Strategy Performance')
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'  # Stock ticker
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    window = 20
    breakout_threshold = 2
    initial_capital = 10000

    # Get historical stock data
    stock_data = get_stock_data(ticker, start=start_date, end=end_date)
    
    # Generate signals based on breakout strategy
    stock_data_with_signals = generate_signals(stock_data, window=window, breakout_threshold=breakout_threshold)
    
    # Backtest the strategy
    backtested_data, final_portfolio_value = backtest_strategy(stock_data_with_signals, initial_capital=initial_capital)
    
    # Output the final portfolio value
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    
    # Plot performance
    plot_performance(backtested_data)
