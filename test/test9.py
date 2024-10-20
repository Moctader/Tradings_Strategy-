import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ffn  # Import ffn for performance analysis

# Step 1: Download Apple Stock Data
aapl_data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
aapl_data.dropna(inplace=True)

# Step 2: Create Moving Averages and RSI
def calculate_technical_indicators(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data

aapl_data = calculate_technical_indicators(aapl_data)

# Step 3: Define Buy, Sell, and Exit Signals
def generate_signals(data):
    data['Buy_Signal'] = ((data['MA50'] > data['MA200']) & (data['RSI'] < 30)).astype(int)
    data['Sell_Signal'] = ((data['MA50'] < data['MA200']) | (data['RSI'] > 70)).astype(int)
    return data

aapl_data = generate_signals(aapl_data)

# Step 4: Strategy Class
class Strategy:
    def __init__(self, buy_signal, sell_signal):
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal
        self.position = 0  # 0 means no position, 1 means long position
        self.positions = []  # To track the positions

    def execute(self, data):
        for i in range(len(data)):
            if self.buy_signal[i] and self.position == 0:
                self.positions.append((i, 'Buy', data['Close'][i]))
                print(f"Buy at {data['Close'][i]} on {data.index[i]}")
                self.position = 1  # Take a long position
            elif self.sell_signal[i] and self.position == 1:
                self.positions.append((i, 'Sell', data['Close'][i]))
                print(f"Sell at {data['Close'][i]} on {data.index[i]}")
                self.position = 0  # Exit the position

# Step 5: Execute the Strategy
buy_signal = aapl_data['Buy_Signal'].values
sell_signal = aapl_data['Sell_Signal'].values
strategy = Strategy(buy_signal, sell_signal)
strategy.execute(aapl_data)

# Step 6: Calculate Portfolio Returns and Backtest using ffn

# Create a dataframe to track portfolio returns
portfolio = pd.DataFrame(index=aapl_data.index, data={'Close': aapl_data['Close']})

# Start with initial cash and no holdings
initial_balance = 10000
portfolio['Signal'] = 0  # 1 = long, 0 = out of market

for position in strategy.positions:
    if position[1] == 'Buy':
        portfolio.loc[portfolio.index[position[0]], 'Signal'] = 1  # Buy signal
    elif position[1] == 'Sell':
        portfolio.loc[portfolio.index[position[0]], 'Signal'] = 0  # Sell signal (exit market)

# Forward-fill the positions
portfolio['Signal'] = portfolio['Signal'].ffill().fillna(0)

# Calculate daily returns and strategy returns
portfolio['Market Returns'] = portfolio['Close'].pct_change()
portfolio['Strategy Returns'] = portfolio['Market Returns'] * portfolio['Signal']

# Step 7: Analyze performance with ffn
strategy_returns = portfolio['Strategy Returns']

# Create performance stats using ffn
stats = ffn.calc_stats(strategy_returns.cumsum())

# Display the performance summary
print(stats.display())

# --- Optional: Plot performance ---
stats.plot()
plt.show()

# --- Optional: You can also save the stats ---
stats.to_csv('performance_summary.csv')

