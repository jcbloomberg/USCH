import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch historical data for S&P 500 and VIX
sp500 = yf.download('^GSPC', start='2000-01-01', end='2024-01-01')
vix = yf.download('^VIX', start='2000-01-01', end='2024-01-01')

# Calculate 30-day moving average and standard deviation
sp500['30_MA'] = sp500['Close'].rolling(window=30).mean()
sp500['30_STD'] = sp500['Close'].rolling(window=30).std()

# Compute Bollinger Bands
sp500['Upper_BB'] = sp500['30_MA'] + 2 * sp500['30_STD']
sp500['Lower_BB'] = sp500['30_MA'] - 2 * sp500['30_STD']

# Identify lower lows in S&P 500
sp500['Lower_Low'] = (sp500['Low'] < sp500['Low'].shift(1)) & (sp500['Low'].shift(1) < sp500['Low'].shift(2))

# Identify VIX spikes
vix['VIX_Spike'] = vix['Close'] > vix['Close'].rolling(window=5).max().shift(1)

# Merge datasets
merged = sp500[['Close', '30_MA', 'Upper_BB', 'Lower_BB', 'Lower_Low']].merge(
    vix[['Close', 'VIX_Spike']], left_index=True, right_index=True, suffixes=('_SP500', '_VIX'))

# Fix column name reference issue
merged.rename(columns={'Close_SP500': 'Close_SP500', 'Close_VIX': 'Close_VIX'}, inplace=True)

# Ensure correct column names after merge
merged.rename(columns={'Close': 'Close_SP500'}, inplace=True)

# Define entry conditions
merged['Entry'] = (merged['Lower_Low']) & (merged['VIX_Spike'].shift(1) & ~merged['VIX_Spike'])
merged['Entry'] &= merged['Close_SP500'] < merged['Upper_BB']

# Define exit conditions
merged['Exit'] = (merged['Close_SP500'] > merged['30_MA']) | (merged['Close_SP500'] < merged['Lower_BB'])

# Backtesting
capital = 100000  # Starting capital
position = 0
returns = []

for i in range(len(merged)):
    if merged.iloc[i]['Entry'] and position == 0:
        position = capital / merged.iloc[i]['Close_SP500']  # Buy S&P 500
        capital = 0
    elif merged.iloc[i]['Exit'] and position > 0:
        capital = position * merged.iloc[i]['Close_SP500']  # Sell S&P 500
        position = 0
    returns.append(capital + (position * merged.iloc[i]['Close_SP500']))

# Convert returns to DataFrame
merged['Portfolio_Value'] = returns

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(merged.index, merged['Portfolio_Value'], label='Portfolio Value', color='blue')
plt.legend()
plt.title('Backtest of S&P 500 - VIX Trading Strategy')
plt.show()
