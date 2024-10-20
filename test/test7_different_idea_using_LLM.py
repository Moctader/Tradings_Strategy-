import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import ffn
import openai
import yaml

# Load context from YAML file
with open('context.yaml', 'r') as file:
    context = yaml.safe_load(file)

# Set up OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

def query_llm(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Step 1: Fetch historical stock data
symbol = context['stock_data']['symbol']
data = yf.download(symbol, start=context['stock_data']['start_date'], end=context['stock_data']['end_date'])
data = data[context['stock_data']['attributes'][:1]]  # Only 'Close' attribute

# Step 2: Prepare data for regression
data['Next_Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Define features and target for regression
X = data[['Close']]  # Current Close price
y = data['Next_Close']  # Next day's Close price

# Step 3: Train Regression Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=context['stock_data']['model_parameters']['regression']['test_size'], 
    random_state=context['stock_data']['model_parameters']['regression']['random_state']
)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 4: Make Predictions
data['Predicted_Close'] = regressor.predict(X)

# Step 5: Prepare data for classification
data['Price_Change'] = data['Predicted_Close'] - data['Close']
data['Signal'] = np.where(data['Price_Change'] > context['stock_data']['signals']['buy_threshold'], 1, 0)  # Buy signal
data['Signal'] = np.where(data['Price_Change'] < context['stock_data']['signals']['sell_threshold'], -1, data['Signal'])  # Sell signal
data['Signal'] = np.where(data['Price_Change'].between(*context['stock_data']['signals']['hold_threshold']), 0, data['Signal'])  # Hold signal

# Remove NaN values created during signal calculation
data.dropna(inplace=True)

# Step 6: Prepare data for classification model
X_classification = data[['Close', 'Predicted_Close', 'Price_Change']]
y_classification = data['Signal']

# Step 7: Train Classification Model
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification, y_classification, test_size=context['stock_data']['model_parameters']['classification']['test_size'], 
    random_state=context['stock_data']['model_parameters']['classification']['random_state']
)
classifier = RandomForestClassifier(random_state=context['stock_data']['model_parameters']['classification']['random_state'])
classifier.fit(X_train_class, y_train_class)

# Step 8: Make Classification Predictions
data['Classified_Signal'] = classifier.predict(X_classification)

# Step 9: Evaluate the classification model
print(classification_report(y_test_class, classifier.predict(X_test_class), target_names=['Hold', 'Buy', 'Sell']))

# Step 10: Evaluate Strategy
data['Daily_Return'] = data['Close'].pct_change()
data['Strategy_Return'] = data['Daily_Return'] * data['Classified_Signal'].shift(1)  # Apply strategy
data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod() - 1

# Print the resulting DataFrame
print(data[['Close', 'Predicted_Close', 'Price_Change', 'Classified_Signal', 'Cumulative_Strategy_Return']])

# Step 11: Use ffn to analyze performance
# Convert to a price series for ffn
portfolio = pd.Series(data['Cumulative_Strategy_Return'], index=data.index)
performance = ffn.Performance(portfolio)

# Display performance metrics
print(performance)

# Step 12: Decision Synthesis and Validation
def validate_strategy(strategy_returns):
    """
    Validate the strategy by calculating various performance metrics.
    """
    cumulative_return = (1 + strategy_returns).cumprod() - 1
    annual_return = cumulative_return.iloc[-1] / (len(strategy_returns) / 252)  # Annualized return
    volatility = strategy_returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    print(f"Cumulative Return: {cumulative_return.iloc[-1]:.2f}")
    print(f"Annualized Return: {annual_return:.2f}")
    print(f"Volatility: {volatility:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Validate the trading strategy
validate_strategy(data['Strategy_Return'])

# Step 13: Use LLM to interpret complex data attributes
llm_prompt = f"""
Given the following stock data attributes and their values:
{data[['Close', 'Predicted_Close', 'Price_Change', 'Classified_Signal', 'Cumulative_Strategy_Return']].tail().to_string(index=False)}

Provide insights and potential strategies based on this data.
"""

llm_insights = query_llm(llm_prompt)
print("LLM Insights and Potential Strategies:")
print(llm_insights)