import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import datetime
import os
import time

TICKER = "SPY"
CSV_FILE = f"{TICKER}.csv"
END = datetime.datetime.today()
START = END - datetime.timedelta(days=60)
START_BALANCE = 100
POSITION_SIZE = 1.0

def download_data(ticker, start, end, retries=3, delay=3):
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1}: Downloading {ticker} data...")
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
    print(f"Deleted old cache file: {CSV_FILE}")

df = download_data(TICKER, START.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d"))
df.to_csv(CSV_FILE)
print("Downloaded and cached new data.")

df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
rsi_gain = df['Close'].diff().clip(lower=0).rolling(14).mean()
rsi_loss = -df['Close'].diff().clip(upper=0).rolling(14).mean()
df['RSI'] = 100 - 100 / (1 + rsi_gain / rsi_loss)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df.dropna(inplace=True)

if df.empty or df.shape[0] < 5:
    raise ValueError("Not enough data after indicator calculations. Increase your lookback period.")

df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

features = ['SMA_10', 'SMA_50', 'RSI', 'MACD']
X = df[features]
y = df['Target']

model = GradientBoostingClassifier()
model.fit(X, y)
df['Pred'] = model.predict(X)

df['Market_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Pred'].shift(1) * df['Market_Return']

account = START_BALANCE
account_curve = [account]

for ret in df['Strategy_Return'].fillna(0):
    trade_amount = account * POSITION_SIZE
    account += trade_amount * ret
    account_curve.append(account)

df['Account_Value'] = account_curve[:-1]
df['BuyHold_Value'] = START_BALANCE * (1 + df['Market_Return']).cumprod()

df['Trade'] = df['Pred'].shift(1) != df['Pred'].shift(2)
df['Trade'] = df['Trade'] & df['Pred'].shift(1).notna()
df['Trade_Date'] = df.index.date
trade_counts = df[df['Trade']]['Trade_Date'].value_counts()
total_trades = trade_counts.sum()
avg_trades_per_day = trade_counts.mean()
days_with_2_or_more_trades = (trade_counts >= 2).sum()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Account_Value'], label='ML Strategy ($)', linewidth=2)
plt.plot(df.index, df['BuyHold_Value'], label='Buy & Hold ($)', linestyle='--')

buy_signals = df[df['Pred'].shift(1) == 1]
plt.scatter(
    buy_signals.index,
    df.loc[buy_signals.index, 'Account_Value'],
    marker='^',
    color='green',
    s=80,
    label='Buy Signal'
)

plt.title(f"{TICKER} | ${START_BALANCE} Backtest: ML Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Account Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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
print(f"Days with 2+ Trades: {days_with_2_or_more_trades}")
