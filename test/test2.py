import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ffn

# 1. Data Collection
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# 2. Feature Engineering
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
stock_data['Momentum'] = stock_data['Close'].diff(1)
stock_data = stock_data.dropna()

# 3. Define Target
stock_data['Target'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)

# 4. Train ML Model
features = ['SMA_50', 'SMA_200', 'Momentum']
X = stock_data[features]
y = stock_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 5. Generate Signals
stock_data['Signal'] = 0
stock_data.loc[X_test.index, 'Signal'] = y_pred
stock_data['Buy_Sell_Signal'] = stock_data['Signal'].diff()

# 6. Backtest Strategy
capital = 10000
buy_signals = stock_data[stock_data['Buy_Sell_Signal'] == 1].index
sell_signals = stock_data[stock_data['Buy_Sell_Signal'] == -1].index
for i in range(len(buy_signals)):
    buy_price = stock_data.loc[buy_signals[i], 'Close']
    sell_price = stock_data.loc[sell_signals[i], 'Close'] if i < len(sell_signals) else stock_data['Close'].iloc[-1]
    profit = (sell_price - buy_price) / buy_price
    capital *= (1 + profit)
print(f"Final portfolio value: {capital:.2f}")

# 7. ffn Backtest
backtest = stock_data['Close'].to_returns().dropna()
stats = backtest.calc_stats()
stats.display()
