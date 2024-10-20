import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Download Apple Stock Data
aapl_data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
aapl_data.dropna(inplace=True)

# Step 2: Create Moving Averages and RSI
def calculate_technical_indicators(data):
    # 50-day and 200-day Moving Averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # RSI (Relative Strength Index)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data

aapl_data = calculate_technical_indicators(aapl_data)

# Step 3: Define Buy, Sell, and Exit Signals
def generate_signals(data):
    # Buy when MA50 crosses above MA200 and RSI < 30 (oversold)
    data['Buy_Signal'] = ((data['MA50'] > data['MA200']) & (data['RSI'] < 30)).astype(int)

    # Sell when MA50 crosses below MA200 or RSI > 70 (overbought)
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
                # Buy Signal
                self.positions.append((i, 'Buy', data['Close'][i]))
                print(f"Buy at {data['Close'][i]} on {data.index[i]}")
                self.position = 1  # Take a long position
            elif self.sell_signal[i] and self.position == 1:
                # Sell Signal
                self.positions.append((i, 'Sell', data['Close'][i]))
                print(f"Sell at {data['Close'][i]} on {data.index[i]}")
                self.position = 0  # Exit the position

# Step 5: Execute the Strategy
buy_signal = aapl_data['Buy_Signal'].values
sell_signal = aapl_data['Sell_Signal'].values
strategy = Strategy(buy_signal, sell_signal)
strategy.execute(aapl_data)

# Step 6: Backtesting the Strategy and Plotting
# Visualize the stock price and the buy/sell signals

plt.figure(figsize=(14, 8))
plt.plot(aapl_data.index, aapl_data['Close'], label='Apple Stock Price', alpha=0.5)
plt.plot(aapl_data.index, aapl_data['MA50'], label='MA50', alpha=0.75)
plt.plot(aapl_data.index, aapl_data['MA200'], label='MA200', alpha=0.75)

# Plot buy signals
buy_signals = [pos[0] for pos in strategy.positions if pos[1] == 'Buy']
sell_signals = [pos[0] for pos in strategy.positions if pos[1] == 'Sell']

plt.scatter(aapl_data.index[buy_signals], aapl_data['Close'][buy_signals], marker='^', color='g', label='Buy Signal', s=100)
plt.scatter(aapl_data.index[sell_signals], aapl_data['Close'][sell_signals], marker='v', color='r', label='Sell Signal', s=100)

plt.title('Apple Stock Price with Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Performance Evaluation
# Calculate returns based on signals
initial_balance = 10000  # Starting with $10,000
balance = initial_balance
shares = 0
buy_price = 0

for position in strategy.positions:
    if position[1] == 'Buy':
        buy_price = position[2]
        shares = balance // buy_price
        balance -= shares * buy_price
    elif position[1] == 'Sell' and shares > 0:
        sell_price = position[2]
        balance += shares * sell_price
        shares = 0

# Final balance if holding any shares
if shares > 0:
    balance += shares * aapl_data['Close'].iloc[-1]

# Display final balance and profit/loss
profit_loss = balance - initial_balance
print(f"Initial balance: ${initial_balance}")
print(f"Final balance: ${balance:.2f}")
print(f"Total profit/loss: ${profit_loss:.2f}")

