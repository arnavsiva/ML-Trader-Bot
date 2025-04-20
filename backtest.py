import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import datetime
import os
import time

TICKER = "PLTR"
CSV_FILE = f"{TICKER}.csv"
END = datetime.datetime.today()
START = END - datetime.timedelta(days=365)
START_BALANCE = 100
POSITION_SIZE = 1.0

def download_data(ticker, start, end, retries=3, delay=3):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end, interval='1h')
            if not df.empty:
                df.index.name = 'Date'
                return df
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(delay)
    raise ValueError(f"Failed to download data for {ticker} after {retries} attempts.")

if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)

df = download_data(TICKER, START.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d"))
df.to_csv(CSV_FILE)

# Indicators
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
rsi_gain = df['Close'].diff().clip(lower=0).rolling(14).mean()
rsi_loss = -df['Close'].diff().clip(upper=0).rolling(14).mean()
df['RSI'] = 100 - 100 / (1 + rsi_gain / rsi_loss)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['Volatility'] = df['Close'].pct_change().rolling(10).std()
df.dropna(inplace=True)

# Target: predict 3-hour forward move > 0.3%
future_return = df['Close'].shift(-3) / df['Close'] - 1
df['Target'] = np.where(future_return > 0.003, 1, 0)
features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility']
df['Pred'] = np.nan

# Rolling window training
train_window = pd.Timedelta(days=30)
step = pd.Timedelta(hours=5)
current = df.index[0] + train_window

while current + step <= df.index[-1]:
    train_start = current - train_window
    train_data = df[(df.index >= train_start) & (df.index < current)]
    test_data = df[(df.index >= current) & (df.index < current + step)]

    if len(train_data) >= 100 and not test_data.empty:
        model = GradientBoostingClassifier()
        model.fit(train_data[features], train_data['Target'])
        proba = model.predict_proba(test_data[features])[:, 1]
        df.loc[test_data.index, 'Pred'] = (proba > 0.6).astype(int)

    current += step

# Backtest logic
df['Market_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Pred'].shift(1) * df['Market_Return']

account = START_BALANCE
account_curve = [account]
for r in df['Strategy_Return'].fillna(0):
    trade_value = account * POSITION_SIZE
    account += trade_value * r
    account_curve.append(account)

df['Account_Value'] = account_curve[:-1]
df['BuyHold_Value'] = START_BALANCE * (1 + df['Market_Return']).cumprod()

# Trade stats
df['Trade'] = df['Pred'].shift(1) != df['Pred'].shift(2)
df['Trade'] &= df['Pred'].shift(1).notna()
df['Trade_Date'] = df.index.date
trade_counts = df[df['Trade']]['Trade_Date'].value_counts()
total_trades = trade_counts.sum()
avg_trades_per_day = trade_counts.mean()
days_with_2_or_more_trades = (trade_counts >= 2).sum()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Account_Value'], label='ML Strategy', linewidth=2)
plt.plot(df.index, df['BuyHold_Value'], label='Buy & Hold', linestyle='--')
buy_signals = df[df['Pred'].shift(1) == 1]
plt.scatter(buy_signals.index, df.loc[buy_signals.index, 'Account_Value'], marker='^', color='green', s=80, label='Buy Signal')
plt.title(f"{TICKER} Backtest: ML Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Account Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final stats
final_ml_value = df['Account_Value'].iloc[-1]
final_bh_value = df['BuyHold_Value'].iloc[-1]
ml_total_return = (final_ml_value - START_BALANCE) / START_BALANCE
bh_total_return = (final_bh_value - START_BALANCE) / START_BALANCE
win_rate = (df['Strategy_Return'] > 0).sum() / df['Strategy_Return'].count()

print(f"ML Final Value: ${final_ml_value:,.2f}")
print(f"Buy & Hold Final Value: ${final_bh_value:,.2f}")
print(f"Total ML Strategy Return: {ml_total_return:.2%}")
print(f"Total Buy & Hold Return: {bh_total_return:.2%}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Total Trades: {total_trades}")
print(f"Avg Trades Per Day: {avg_trades_per_day:.2f}")
print(f"Days w/ 2+ Trades: {days_with_2_or_more_trades}")
