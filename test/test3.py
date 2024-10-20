import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from abc import ABC, abstractmethod

# 1. Define the Signal class (Abstract Class)

class Signal(ABC):
    @abstractmethod
    def generate_signals(self, data):
        """Generate buy and sell signals based on the data"""
        pass

class ConfidenceSignal(Signal):
    def __init__(self, model):
        self.model = model
    
    def generate_signals(self, data):
        # Generate features
        X = data[['SMA_50', 'SMA_200', 'Momentum']]
        
        # Predict signal probabilities (buy, hold, sell)
        probabilities = self.model.predict_proba(X)
        data['Signal'] = self.model.predict(X)  # 1 for buy, -1 for sell
        
        # Capture the model's confidence (max probability between buy or sell)
        data['Confidence'] = probabilities.max(axis=1)  # Highest confidence for buy/sell
        return data

# 2. Define the Strategy class (Abstract Class)

class Strategy(ABC):
    @abstractmethod
    def execute(self, data):
        """Run strategy based on signals"""
        pass

class ConfidenceBasedStrategy(Strategy):
    def __init__(self, initial_capital=10000, max_position_size=0.2):
        self.capital = initial_capital
        self.positions = 0  # Track the number of units held
        self.cash = initial_capital
        self.max_position_size = max_position_size  # Max 20% of the portfolio in any one trade
    
    def execute(self, data):
        portfolio_value = self.capital
        
        for i in range(1, len(data)):
            signal = data['Signal'].iloc[i]
            confidence = data['Confidence'].iloc[i]
            price = data['Close'].iloc[i]
            
            # Calculate position size based on confidence (higher confidence -> bigger position)
            position_size = confidence * self.max_position_size * portfolio_value // price
            
            # Ensure we do not exceed the maximum position size
            position_size = min(position_size, self.cash // price)
            
            # Buy signal (confidence-based)
            if signal == 1 and position_size > 0:
                self.positions += position_size
                self.cash -= position_size * price
                print(f"Bought {position_size} units at {price} with confidence {confidence:.2f}")
                
            # Sell signal (confidence-based)
            elif signal == -1 and self.positions > 0:
                sell_units = min(self.positions, position_size)
                self.positions -= sell_units
                self.cash += sell_units * price
                print(f"Sold {sell_units} units at {price} with confidence {confidence:.2f}")
        
        # Final portfolio value (cash + value of remaining positions)
        final_value = self.cash + self.positions * data['Close'].iloc[-1]
        print(f"Final portfolio value: {final_value:.2f}")
        return final_value

# 3. Collect and Preprocess Data
def fetch_stock_data(stock, start, end):
    data = yf.download(stock, start=start, end=end)
    # Add technical indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Momentum'] = data['Close'].diff(1)
    return data.dropna()

# 4. Train ML Model
def train_model(data):
    # Define features and target
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
    
    features = ['SMA_50', 'SMA_200', 'Momentum']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

# 5. Run the Simulation
def run_backtest(stock, start, end, strategy):
    # Fetch stock data
    data = fetch_stock_data(stock, start, end)
    
    # Train a model to generate signals
    model = train_model(data)
    
    # Create signal generator with confidence-based signals
    signal_generator = ConfidenceSignal(model)
    data = signal_generator.generate_signals(data)
    
    # Execute strategy
    final_value = strategy.execute(data)
    print(f"Final portfolio value: {final_value:.2f}")

# Running the complete implementation with confidence-based position sizing

strategy = ConfidenceBasedStrategy(initial_capital=10000, max_position_size=0.2)
run_backtest('AAPL', '2020-01-01', '2023-01-01', strategy)
