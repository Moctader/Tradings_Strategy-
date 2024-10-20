import backtrader as bt
import yfinance as yf
import datetime
import ffn  # for financial performance metrics
import pandas as pd
import backtrader.analyzers as btanalyzers

# Step 2: Create a Trading Strategy
class MovingAverageCrossStrategy(bt.Strategy):
    params = (
        ("short_window", 40),
        ("long_window", 100),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.short_window
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.long_window
        )
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if self.crossover > 0:  # if short MA crosses above long MA
            print(f"Buy signal on {self.data.datetime.date(0)} at price {self.data_close[0]:.2f}")
            self.buy()
        elif self.crossover < 0:  # if short MA crosses below long MA
            print(f"Sell signal on {self.data.datetime.date(0)} at price {self.data_close[0]:.2f}")
            self.sell()

# Custom observer to record portfolio value for every trading day
class PortfolioValue(bt.Observer):
    lines = ('value',)
    plotinfo = dict(plot=True, subplot=True)

    def next(self):
        # Record portfolio value at each step (daily or as defined by data frequency)
        self.lines.value[0] = self._owner.broker.getvalue()
        # Debugging: Print the recorded portfolio value for each day
        print(f"Portfolio Value on {self.datas[0].datetime.date(0)}: {self.lines.value[0]:.2f}")

# Step 3: Load Historical Data using yfinance
data_df = yf.download('AAPL', start='2020-01-01', end='2022-01-01')
data = bt.feeds.PandasData(dataname=data_df)

# Step 4: Set Up the Backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MovingAverageCrossStrategy)
cerebro.adddata(data)
cerebro.broker.set_cash(100000)  # Start with 100,000 cash

# Add the TimeReturn analyzer to track returns over time
cerebro.addanalyzer(btanalyzers.TimeReturn, _name='timereturn')

# Add custom observer to track portfolio value
cerebro.addobserver(PortfolioValue)

# Step 5: Add Additional Settings (Optional)
cerebro.addsizer(bt.sizers.FixedSize, stake=1000)
cerebro.broker.setcommission(commission=0.001)  # 0.1% commission (example)

# Step 6: Capture Starting Value and Run the Backtest
starting_value = cerebro.broker.getvalue()
print(f'Starting Portfolio Value: {starting_value:.2f}')

# Run the backtest and collect analyzers
results = cerebro.run()
timereturn = results[0].analyzers.timereturn.get_analysis()

# Step 7: Capture Ending Value
ending_value = cerebro.broker.getvalue()
print(f'Ending Portfolio Value: {ending_value:.2f}')

# Step 8: Convert returns to a pandas Series for analysis
returns_series = pd.Series(timereturn)
print(f"Returns series: {returns_series}")

# Ensure that the length of portfolio_values matches the length of the data index
if len(returns_series) < len(data_df.index):
    data_df = data_df.iloc[:len(returns_series)]

# Step 9: Analyze Performance Using ffn
perf_stats = returns_series.calc_stats()

# Step 10: Print and Display Performance Metrics
print("Performance Metrics:")
print(f"Total Return: {perf_stats.stats['total_return'] * 100:.2f}%")
print(f"Annual Return (CAGR): {perf_stats.stats['cagr'] * 100:.2f}%")
print(f"Max Drawdown: {perf_stats.stats['max_drawdown'] * 100:.2f}%")
print(f"Sharpe Ratio: {perf_stats.stats['daily_sharpe']:.2f}")
print(f"Sortino Ratio: {perf_stats.stats['daily_sortino']:.2f}")
#print(f"Win Ratio (Monthly Positive %): {perf_stats.stats['pos_month_perc'] * 100:.2f}%")

# Step 11: Visualize the Results
cerebro.plot()
