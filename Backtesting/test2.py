import backtrader as bt
import yfinance as yf
import datetime

# Create a Strategy
class BollingerRSIStrategy(bt.Strategy):
    params = (
        ("bband_period", 20),
        ("rsi_period", 14),
        ("overbought", 70),
        ("oversold", 30),
        ("stddev", 2),
    )

    def __init__(self):
        self.bband = bt.indicators.BollingerBands(
            self.data.close, period=self.params.bband_period, devfactor=self.params.stddev
        )
        self.rsi = bt.indicators.RelativeStrengthIndex(
            period=self.params.rsi_period
        )

    def next(self):
        if self.data.close < self.bband.lines.bot and self.rsi < self.params.oversold:
            self.buy()
        elif self.data.close > self.bband.lines.top and self.rsi > self.params.overbought:
            self.sell()

# Download historical data
data_df = yf.download('AAPL', start='2020-01-01', end='2022-01-01')
data = bt.feeds.PandasData(dataname=data_df)

# Create a Cerebro entity
cerebro = bt.Cerebro(stdstats=False)

# Add a strategy
cerebro.addstrategy(BollingerRSIStrategy)

# Add the Data Feed to Cerebro
cerebro.adddata(data)

# Set our desired cash start
cerebro.broker.set_cash(100000.0)

# Add a FixedSize sizer according to the stake
cerebro.addsizer(bt.sizers.FixedSize, stake=1000)

# Set the commission
cerebro.broker.setcommission(commission=0.001)

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Plot the result
cerebro.plot(iplot=True)
