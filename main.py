import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
import discord
from discord.ext import commands, tasks
import datetime
import os
import matplotlib.pyplot as plt
import pytz
import asyncio
from dotenv import load_dotenv
import json
import joblib

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TICKER = "SPY"
ACCOUNT_FILE = "account.csv"
TRADE_LOG = "trades.csv"
MODEL_FILE = "model.pkl"
DATASET_FILE = "dataset.csv"
START_BALANCE = 100
POSITION_SIZE = 1.0
CHANNEL_ID = 1363325653096726589
TIMEZONE = pytz.timezone("America/Chicago")
START_TIME = TIMEZONE.localize(datetime.datetime(2025, 4, 22, 8, 30))

intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)

async def update_bot_status():
    trades = pd.read_csv(TRADE_LOG)
    acct = load_account()
    if trades.empty:
        trade_count = 0
        pnl = 0.0
    else:
        trades['cash_value'] = trades.apply(
            lambda row: row['shares'] * row['price'] if row['action'] == 'SELL' else -row['shares'] * row['price'],
            axis=1
        )
        trade_count = len(trades)
        pnl = trades['cash_value'].sum()

    days_running = (datetime.datetime.now(TIMEZONE) - START_TIME).days
    status_msg = f"📈 Watching {TICKER} | {days_running}d | {trade_count} trades | Net ${pnl:.2f}"
    await bot.change_presence(activity=discord.Game(name=status_msg))

@tasks.loop(minutes=60)
async def update_status_loop():
    await update_bot_status()

@tasks.loop(time=datetime.time(hour=15, minute=0, tzinfo=TIMEZONE))
async def append_daily_data():
    print("[INFO] Appending daily data...")
    df_new = fetch_data(period="1d")
    if os.path.exists(DATASET_FILE):
        df_old = pd.read_csv(DATASET_FILE, parse_dates=["Datetime"])
        df_new.reset_index(inplace=True)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df.drop_duplicates(subset="Datetime", keep="last", inplace=True)
        df = df[df["Datetime"] > (datetime.datetime.now() - datetime.timedelta(days=730))]
    else:
        df_new.reset_index(inplace=True)
        df_new = df_new.rename(columns={"index": "Datetime"})
        df = df_new
    df.to_csv(DATASET_FILE, index=False)
    print("[INFO] Daily data appended.")

    print("[INFO] Retraining model...")
    df = pd.read_csv(DATASET_FILE, parse_dates=["Datetime"])
    future_return = df['Close'].shift(-3) / df['Close'] - 1
    df['Target'] = np.where(future_return > 0.003, 1, 0)
    df.dropna(inplace=True)
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility']
    X = df[features]
    y = df['Target']
    model = GradientBoostingClassifier()
    model.fit(X, y)
    joblib.dump((model, features), MODEL_FILE)
    print("[INFO] Model retrained and saved.")

def load_holidays():
    try:
        with open("holidays.json", "r") as file:
            return set(datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in json.load(file))
    except FileNotFoundError:
        return set()

def init_account():
    if not os.path.exists(ACCOUNT_FILE):
        pd.DataFrame([{
            "cash": START_BALANCE,
            "shares": 0,
            "pending_settlement": 0.0,
            "settlement_date": None
        }]).to_csv(ACCOUNT_FILE, index=False)
    if not os.path.exists(TRADE_LOG):
        pd.DataFrame(columns=["timestamp", "action", "price", "shares", "reason"]).to_csv(TRADE_LOG, index=False)

def load_account():
    return pd.read_csv(ACCOUNT_FILE).iloc[0]

def save_account(cash, shares, pending_settlement, settlement_date):
    pd.DataFrame([{
        "cash": cash,
        "shares": shares,
        "pending_settlement": pending_settlement,
        "settlement_date": settlement_date
    }]).to_csv(ACCOUNT_FILE, index=False)

def fetch_data(period="15d"):
    df = yf.download(TICKER, period=period, interval="1h")
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    rsi_gain = df['Close'].diff().clip(lower=0).rolling(14).mean()
    rsi_loss = -df['Close'].diff().clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + rsi_gain / rsi_loss)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    df.dropna(inplace=True)
    return df

def train_initial_model():
    df = fetch_data(period="729d")
    df.reset_index(inplace=True)
    df.to_csv(DATASET_FILE, index=False)
    future_return = df['Close'].shift(-3) / df['Close'] - 1
    df['Target'] = np.where(future_return > 0.003, 1, 0)
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility']
    X = df[features]
    y = df['Target']
    model = GradientBoostingClassifier()
    model.fit(X, y)
    joblib.dump((model, features), MODEL_FILE)
    return model, features

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_initial_model()

def generate_signal(model, latest_row, features):
    values = latest_row[features].values.reshape(1, -1)
    proba = model.predict_proba(values)[0, 1]
    return int(proba > 0.6), proba

def execute_trade(signal, price, confidence):
    acct = load_account()
    cash = acct['cash']
    shares = acct['shares']
    pending_settlement = acct['pending_settlement']
    settlement_date = acct['settlement_date']
    action, executed_shares, reason = None, 0.0, ""

    if signal == 1 and cash >= price:
        executed_shares = round(cash / price, 4)
        if executed_shares > 0:
            cash -= executed_shares * price
            shares += executed_shares
            action = "BUY"
            reason = f"Model predicted price increase with {confidence:.2%} confidence."
    elif signal == 0 and shares > 0:
        executed_shares = shares
        proceeds = shares * price
        pending_settlement += proceeds
        settlement_date = (datetime.datetime.now(TIMEZONE) + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        shares = 0
        action = "SELL"
        reason = f"Model predicted price drop with {confidence:.2%} confidence."

    if action:
        save_account(cash, shares, pending_settlement, settlement_date)
        pd.DataFrame([{
            "timestamp": datetime.datetime.now(),
            "action": action,
            "price": price,
            "shares": executed_shares,
            "reason": reason
        }]).to_csv(TRADE_LOG, mode='a', header=False, index=False)
        return action, executed_shares, price, cash + shares * price + pending_settlement, reason
    return None, 0, price, cash + shares * price + pending_settlement, "Model did not predict a buy or sell opportunity."

@tasks.loop(minutes=1)
async def check_settlement():
    now = datetime.datetime.now(TIMEZONE)
    if now.hour == 8 and now.minute == 0:
        acct = load_account()
        if acct['settlement_date'] and acct['settlement_date'] <= now.strftime("%Y-%m-%d"):
            new_cash = acct['cash'] + acct['pending_settlement']
            save_account(new_cash, acct['shares'], 0.0, None)
            channel = bot.get_channel(CHANNEL_ID)
            await channel.send(f"💰 ${acct['pending_settlement']:.2f} settled and moved to available cash.")

@bot.command()
async def account(ctx):
    acct = load_account()
    last_price = yf.Ticker(TICKER).history(period="1d")['Close'].iloc[-1]
    total_value = acct['cash'] + acct['shares'] * last_price + acct['pending_settlement']
    await ctx.send(
        f"Available Cash: ${acct['cash']:.2f}\n"
        f"Pending Settlement: ${acct['pending_settlement']:.2f}\n"
        f"Shares: {int(acct['shares'])}\n"
        f"Total Account Value: ${total_value:.2f}"
    )

@bot.command()
async def trade(ctx):
    await ctx.send("Checking for signal...")
    df = fetch_data()
    model, features = load_model()
    signal, confidence = generate_signal(model, df.iloc[-1], features)
    price = df.iloc[-1]['Close']
    action, shares, price, value, reason = execute_trade(signal, price, confidence)
    if action:
        await ctx.send(f"{action} {shares:.4f} shares of {TICKER} at ${price:.2f}\nConfidence: {confidence:.2%}\nReason: {reason}\nAccount Value: ${value:.2f}")
    else:
        await ctx.send(f"No trade executed. {reason}")

@bot.command()
async def graph(ctx):
    acct = load_account()
    trades = pd.read_csv(TRADE_LOG)
    if trades.empty:
        await ctx.send("No trades to show.")
        return

    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    trades['cash_value'] = trades.apply(lambda row: row['shares'] * row['price'] if row['action'] == 'SELL' else -row['shares'] * row['price'], axis=1)
    trades['equity'] = trades['cash_value'].cumsum() + START_BALANCE

    plt.figure(figsize=(10,5))
    plt.plot(trades['timestamp'], trades['equity'], label='Account Value')
    plt.title('Account Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('USD')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('account_value.png')
    plt.close()

    await ctx.send(file=discord.File('account_value.png'))

@bot.command()
async def strat(ctx):
    explanation = (
        "**📊 Strategy Explanation:**\n"
        f"This bot uses a machine learning model (Gradient Boosting Classifier) to {TICKER} trade during U.S. market hours.\n\n"
        "**✅ Trading Rules:**\n"
        "- **Runs only on weekdays** (Monday–Friday).\n"
        "- **Avoids U.S. market holidays** listed in `holidays.json`.\n"
        "- **Only trades between 8:30 AM and 3:00 PM CST.**\n\n"
        "**📈 Strategy Details:**\n"
        f"- Pulls last 15 days of hourly {TICKER} data.\n"
        "- Calculates indicators: SMA_10, SMA_50, RSI, MACD, Volatility.\n"
        f"- Predicts if {TICKER} will rise over the next 3 hours by >0.3%.\n"
        "- If confidence > 60%, places a trade using available cash or existing position.\n"
        "- Trades fractional shares. Sells fully. Settles proceeds next day.\n\n"
        "Use `!account`, `!graph`, or `!trade` to interact with the bot."
    )
    await ctx.send(explanation)

@tasks.loop(minutes=1)
async def scheduled_trade():
    now = datetime.datetime.now(TIMEZONE)
    holidays = load_holidays()

    if now.weekday() >= 5:
        return

    if now.date() in holidays:
        channel = bot.get_channel(CHANNEL_ID)
        await channel.send(f"⛔ Market is closed today ({now.date()}) due to a holiday. Skipping trades.")
        return

    market_open = now.replace(hour=8, minute=30)
    market_close = now.replace(hour=15, minute=0)

    if now < market_open or now > market_close:
        return

    if now.minute == 31:
        channel = bot.get_channel(CHANNEL_ID)
        await channel.send(f"🕐 Hourly check at {now.strftime('%I:%M %p')} CST")
        await run_trade(channel)

async def run_trade(channel):
    df = fetch_data()
    model, features = load_model()
    signal, confidence = generate_signal(model, df.iloc[-1], features)
    price = df.iloc[-1]['Close']
    action, shares, price, value, reason = execute_trade(signal, price, confidence)
    if action:
        await channel.send(f"{action} {shares:.4f} shares of {TICKER} at ${price:.2f}\nConfidence: {confidence:.2%}\nReason: {reason}\nAccount Value: ${value:.2f}")
    else:
        await channel.send(f"No trade executed. {reason}")

@bot.command()
async def test(ctx):
    try:
        df = pd.read_csv(DATASET_FILE)

        for col in ['Close', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        if 'Target' not in df.columns:
            df['FutureReturn'] = df['Close'].shift(-3) / df['Close'] - 1
            df['Target'] = np.where(df['FutureReturn'] > 0.003, 1, 0)
            df.drop(columns=['FutureReturn'], inplace=True)
            df.dropna(inplace=True)

        model, features = load_model()
        latest_df = fetch_data()
        latest_row = latest_df.iloc[-1]
        values = latest_row[features].astype(float).values.reshape(1, -1)
        proba = model.predict_proba(values)[0, 1]
        if proba > 0.6:
            prediction = "UP 📈"
            confidence = proba
        else:
            prediction = "DOWN 📉"
            confidence = 1 - proba

        expected_up_means = df[df['Target'] == 1][features].mean()
        expected_up_std = df[df['Target'] == 1][features].std()

        indicators = []
        for feat in features:
            current = latest_row[feat].item()
            expected = expected_up_means[feat]
            std = expected_up_std[feat]

            if std == 0 or np.isnan(std):
                deviation = 0
            else:
                deviation = abs(current - expected) / std

            if deviation < 0.5:
                emoji = "✅"
                arrow = "⬆️"
            elif deviation < 1.0:
                emoji = "⚠️"
                arrow = "↔"
            else:
                emoji = "❌"
                arrow = "⬇️"

            indicators.append(
                f"{emoji} **{feat}**: {current:.4f} | expected: {expected:.4f} {arrow}"
            )

        prediction_vals = (
            f"Class 0 (↓): {1 - proba:.2%}\n"
            f"Class 1 (↑): {proba:.2%}"
        )

        await ctx.send(
            f"**🔎 Current vs Expected Indicators:**\n" +
            "\n".join(indicators) +
            f"\n\n**📊 Model Prediction Probabilities:**\n{prediction_vals}\n\n" +
            f"**🧠 Final Signal:** **{prediction}** (Confidence: {confidence:.2%})"
        )

    except Exception as e:
        await ctx.send(f"❌ Error in !test: {e}")

@bot.event
async def on_ready():
    init_account()
    print("[INFO] Bot is initializing model...")
    if not os.path.exists(MODEL_FILE):
        print("[INFO] Training initial model...")
        train_initial_model()
    else:
        print("[INFO] Model found. Skipping training.")

    channel = bot.get_channel(CHANNEL_ID)
    await channel.send("✅ ML bot is successfully online.")
    if not scheduled_trade.is_running():
        scheduled_trade.start()
    if not check_settlement.is_running():
        check_settlement.start()
    if not update_status_loop.is_running():
        update_status_loop.start()
    if not append_daily_data.is_running():
        append_daily_data.start()

async def main():
    await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
