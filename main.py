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

load_dotenv()

# config
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TICKER = "SPY"
ACCOUNT_FILE = "account.csv"
TRADE_LOG = "trades.csv"
START_BALANCE = 100
POSITION_SIZE = 1.0
CHANNEL_ID = 1363325653096726589
TIMEZONE = pytz.timezone("America/Chicago")
START_TIME = TIMEZONE.localize(datetime.datetime(2025, 4, 21, 8, 30))

# init discord bot
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

# init account
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

# download latest data
def fetch_latest_data():
    df = yf.download(TICKER, period="15d", interval="1h")
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    rsi_gain = df['Close'].diff().clip(lower=0).rolling(14).mean()
    rsi_loss = -df['Close'].diff().clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + rsi_gain / rsi_loss)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df.dropna(inplace=True)
    return df

# ml detect signal
def train_model(df):
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD']
    X = df[features]
    y = df['Target']
    model = GradientBoostingClassifier()
    model.fit(X, y)
    return model

def generate_signal(model, latest_row):
    features = latest_row[['SMA_10', 'SMA_50', 'RSI', 'MACD']].values.reshape(1, -1)
    return model.predict(features)[0]

# make trades
def execute_trade(signal, price):
    acct = load_account()
    cash = acct['cash']
    shares = int(acct['shares'])
    pending_settlement = acct['pending_settlement']
    settlement_date = acct['settlement_date']
    action, executed_shares, reason = None, 0, ""

    if signal == 1 and cash >= price:
        executed_shares = int(cash // price)
        if executed_shares > 0:
            cash -= executed_shares * price
            shares += executed_shares
            action = "BUY"
            reason = "Model predicted price increase."
    elif signal == 0 and shares > 0:
        executed_shares = shares
        proceeds = shares * price
        pending_settlement += proceeds
        settlement_date = (datetime.datetime.now(TIMEZONE) + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        shares = 0
        action = "SELL"
        reason = "Model predicted price drop."

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

# check if settlement
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

# discord commands
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
    await run_trade(ctx.channel)

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
        "This bot uses **machine learning** to trade SPY every hour on weekdays.\n\n"
        "**Indicators it uses:**\n"
        "- **SMA_10 vs SMA_50**: Tracks short vs long-term trend.\n"
        "- **RSI**: Measures momentum and overbought/oversold status.\n"
        "- **MACD**: Measures momentum and signal crossovers.\n\n"
        "**How it works:**\n"
        "1. Every hour, it pulls the last 15 days of hourly data.\n"
        "2. It trains a Gradient Boosting model to predict if the next hour will go up.\n"
        "3. If yes and there's available cash, it buys whole shares.\n"
        "4. If no and there are shares, it sells all. Proceeds settle next business day.\n"
        "5. It tracks and separates settled and unsettled cash.\n\n"
        "Track trades with `!account` or `!graph`."
    )
    await ctx.send(explanation)

# trade ml every hour
@tasks.loop(minutes=1)
async def scheduled_trade():
    now = datetime.datetime.now(TIMEZONE)
    if now.weekday() < 5 and now.minute == 0:
        channel = bot.get_channel(CHANNEL_ID)
        await channel.send(f"🕐 Hourly check at {now.strftime('%I:%M %p')} CST")
        await run_trade(channel)

async def run_trade(channel):
    df = fetch_latest_data()
    model = train_model(df)
    signal = generate_signal(model, df.iloc[-1])
    price = df.iloc[-1]['Close']
    action, shares, price, value, reason = execute_trade(signal, price)
    if action:
        await channel.send(f"{action} {shares} shares of {TICKER} at ${price:.2f}\nReason: {reason}\nAccount Value: ${value:.2f}")
    else:
        await channel.send(f"No trade executed. {reason}")

# start bot
@bot.event
async def on_ready():
    init_account()
    channel = bot.get_channel(CHANNEL_ID)
    await channel.send("✅ ML bot is successfully online.")
    if not scheduled_trade.is_running():
        scheduled_trade.start()
    if not check_settlement.is_running():
        check_settlement.start()
    if not update_status_loop.is_running():
        update_status_loop.start()

async def main():
    await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
