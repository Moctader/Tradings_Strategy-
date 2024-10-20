import backtrader as bt
import yfinance as yf
import datetime

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
            self.buy()
        elif self.crossover < 0:  # if short MA crosses below long MA
            self.sell()

# Step 3: Load Historical Data using yfinance
data_df = yf.download('AAPL', start='2020-01-01', end='2022-01-01')
data = bt.feeds.PandasData(dataname=data_df)

# Step 4: Set Up the Backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MovingAverageCrossStrategy)
cerebro.adddata(data)
cerebro.broker.set_cash(100000)  # e.g. start with 100,000 cash

# Step 5: Add Additional Settings (Optional)
cerebro.addsizer(bt.sizers.FixedSize, stake=1000)
cerebro.broker.setcommission(commission=0.001)  # 0.1% commission (example)

# Step 6: Run the Backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Step 7: Visualize the Results
cerebro.plot()