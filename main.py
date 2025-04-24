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
from io import BytesIO
import shap
import shutil
import psutil

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TICKER = "" # add your ticker
ACCOUNT_FILE = "account.csv"
TRADE_LOG = "trades.csv"
MODEL_FILE = "model.pkl"
DATASET_FILE = "dataset.csv"
SETTLEMENT_FILE = "settlement.csv"
START_BALANCE = 100
POSITION_SIZE = 1.0
CHANNEL_ID = 0 # add your channel ID
LOGS_CHANNEL_ID = 0 # add your channel ID
TIMEZONE = pytz.timezone("America/Chicago")
START_TIME = TIMEZONE.localize(datetime.datetime(0, 0, 0, 0, 0)) # add start time (year, month, day, hour minute)

intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)

async def update_bot_status():
    acct = load_account()
    last_price = yf.Ticker(TICKER).history(period="1d")['Close'].iloc[-1]

    pending_settlement = 0.0
    if os.path.exists(SETTLEMENT_FILE):
        df = pd.read_csv(SETTLEMENT_FILE, parse_dates=["date"])
        pending_settlement = df["amount"].sum()

    total_value = acct['cash'] + acct['shares'] * last_price + pending_settlement

    trades = pd.read_csv(TRADE_LOG)
    trade_count = len(trades) if not trades.empty else 0
    days_running = (datetime.datetime.now(TIMEZONE) - START_TIME).days

    status_msg = f"📈 {TICKER} | {days_running}d | {trade_count} trades | ${total_value:.2f}"
    await bot.change_presence(activity=discord.Game(name=status_msg))

@tasks.loop(minutes=60)
async def update_status_loop():
    await update_bot_status()

async def log_to_channel(message: str):
    print(message)
    channel = bot.get_channel(LOGS_CHANNEL_ID)
    if channel:
        await channel.send(f"[BOT LOG] {message}")

@tasks.loop(time=datetime.time(hour=15, minute=0, tzinfo=TIMEZONE))
async def append_daily_data():
    await log_to_channel("Appending daily data...")
    now = datetime.datetime.now(TIMEZONE)
    embed = discord.Embed(
        title="🕒 Appending daily data",
        description=f"⏰ It is 3:00. Market's are closed and the dataset is appending daily data.",
        color=0x00ff99
    )
    embed.timestamp = now
    
    channel = bot.get_channel(CHANNEL_ID)
    await channel.send(embed=embed)
    
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
    await log_to_channel("Daily data appended.")

    await log_to_channel("Retraining model...")
    df = pd.read_csv(DATASET_FILE, parse_dates=["Datetime"])
    future_return = df['Close'].shift(-3) / df['Close'] - 1
    df['Target'] = np.where(future_return > 0.003, 1, 0)
    df.dropna(inplace=True)
    features = [
    'SMA_Ratio', 'RSI', 'ATR', 'OBV',
    'BB_Width', 'Stochastic_K', 'Stochastic_D',
    'Donchian_Mid', 'Donchian_Width'
]
    X = df[features]
    y = df['Target']
    model = GradientBoostingClassifier()
    model.fit(X, y)
    joblib.dump((model, features), MODEL_FILE)
    await log_to_channel("Model retrained and saved.")

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
            "shares": 0
        }]).to_csv(ACCOUNT_FILE, index=False)

    if not os.path.exists(TRADE_LOG):
        pd.DataFrame(columns=["timestamp", "action", "price", "shares", "reason"]).to_csv(TRADE_LOG, index=False)

    if not os.path.exists(SETTLEMENT_FILE):
        pd.DataFrame(columns=["date", "amount"]).to_csv(SETTLEMENT_FILE, index=False)

def load_account():
    acct = pd.read_csv(ACCOUNT_FILE).iloc[0]
    return {
        "cash": float(acct["cash"]),
        "shares": float(acct["shares"])
    }

def save_account(cash, shares):
    pd.DataFrame([{
        "cash": cash,
        "shares": shares
    }]).to_csv(ACCOUNT_FILE, index=False)

def fetch_data(period="15d"):
    df = yf.download(TICKER, period=period, interval="1h")
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_Ratio'] = df['SMA_10'] / df['SMA_50']

    rsi_gain = df['Close'].diff().clip(lower=0).rolling(14).mean()
    rsi_loss = -df['Close'].diff().clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + rsi_gain / rsi_loss)

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR'], inplace=True)

    df['OBV'] = 0
    df.loc[df.index[1:], 'OBV'] = np.where(
    df['Close'].iloc[1:].values > df['Close'].iloc[:-1].values,
    df['Volume'].iloc[1:].values,
    np.where(
        df['Close'].iloc[1:].values < df['Close'].iloc[:-1].values,
        -df['Volume'].iloc[1:].values,
        0
    )
)
    df['OBV'] = df['OBV'].cumsum()

    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

    df['Donchian_High'] = df['High'].rolling(window=20).max()
    df['Donchian_Low'] = df['Low'].rolling(window=20).min()
    df['Donchian_Mid'] = (df['Donchian_High'] + df['Donchian_Low']) / 2
    df['Donchian_Width'] = df['Donchian_High'] - df['Donchian_Low']

    df.dropna(inplace=True)
    return df

async def train_initial_model():
    df = fetch_data(period="729d")
    df.reset_index(inplace=True)
    df.to_csv(DATASET_FILE, index=False)

    future_return = df['Close'].shift(-3) / df['Close'] - 1
    df['Target'] = np.where(future_return > 0.003, 1, 0)

    features = [
    'SMA_Ratio', 'RSI', 'ATR', 'OBV',
    'BB_Width', 'Stochastic_K', 'Stochastic_D',
    'Donchian_Mid', 'Donchian_Width'
]
    X = df[features]
    y = df['Target']

    model = GradientBoostingClassifier()
    model.fit(X, y)

    importances = model.feature_importances_
    ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    await log_to_channel("📊 Feature Importance Ranking:")
    for feat, imp in ranked:
        await log_to_channel(f"{feat}: {imp:.4f}")

    joblib.dump((model, features), MODEL_FILE)
    return model, features

async def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return await train_initial_model()

def generate_signal(model, latest_row, features):
    values = latest_row[features].values.reshape(1, -1)
    proba = model.predict_proba(values)[0, 1]
    return int(proba > 0.6), proba

def execute_trade(signal, price, confidence):
    acct = load_account()
    cash = acct['cash']
    shares = acct['shares']
    pending_settlement = 0.0
    if os.path.exists(SETTLEMENT_FILE):
        df_set = pd.read_csv(SETTLEMENT_FILE, parse_dates=["date"])
        pending_settlement = df_set["amount"].sum()
    account_value = cash + shares * price + pending_settlement

    action, executed_shares, reason = None, 0.0, ""
    if signal == 1 and (cash - 5) > 0:
        executed_shares = round((cash - 5) / price, 4)
        if executed_shares > 0:
            cash -= executed_shares * price
            shares += executed_shares
            action = "BUY"
            reason = f"Model predicted price increase next 3h with {confidence:.2%} confidence."
    elif signal == 0 and shares > 0:
        executed_shares = shares
        proceeds = shares * price
        log_settlement(proceeds)
        cash = cash
        shares = 0
        action = "SELL"
        reason = f"Model predicted price drop next 3h with {confidence:.2%} confidence."

    if action:
        save_account(cash, shares)
        pd.DataFrame([{
            "timestamp": datetime.datetime.now(),
            "action": action,
            "price": price,
            "shares": executed_shares,
            "reason": reason
        }]).to_csv(TRADE_LOG, mode='a', header=False, index=False)
        account_value = cash + shares * price + pending_settlement

        return action, executed_shares, price, account_value, reason

    return None, 0, price, account_value, "No trade action taken."

@tasks.loop(time=datetime.time(hour=8, minute=0, tzinfo=TIMEZONE))
async def check_settlement():
    now = datetime.datetime.now(TIMEZONE)
    today = now.date()
    yesterday = today - datetime.timedelta(days=1)

    if os.path.exists(SETTLEMENT_FILE):
        df = pd.read_csv(SETTLEMENT_FILE, parse_dates=["date"])
        df_to_settle = df[df["date"].dt.date == yesterday]

        if not df_to_settle.empty:
            total_settlement = df_to_settle["amount"].sum()
            acct = load_account()
            new_cash = acct["cash"] + total_settlement

            save_account(new_cash, acct["shares"])

            df = df[df["date"].dt.date != yesterday]
            df.to_csv(SETTLEMENT_FILE, index=False)

            embed = discord.Embed(
                title="💰 Settlement Complete",
                description=f"${total_settlement:.2f} has been settled and added to available cash.",
                color=0x00ff99
            )
            embed.timestamp = now

            channel = bot.get_channel(CHANNEL_ID)
            await channel.send(embed=embed)

def log_settlement(amount):
    now = datetime.datetime.now(TIMEZONE)
    new_entry = pd.DataFrame([{"date": now, "amount": amount}])

    df = pd.DataFrame(columns=["date", "amount"])
    if os.path.exists(SETTLEMENT_FILE):
        df = pd.read_csv(SETTLEMENT_FILE, parse_dates=["date"])
        df = df[["date", "amount"]].dropna(subset=["date", "amount"], how="any")

    if df.empty or df.isna().all().all():
        df = new_entry
    else:
        df = pd.concat([df, new_entry], ignore_index=True)

    df.to_csv(SETTLEMENT_FILE, index=False)

@bot.command()
async def account(ctx):
    """
    Show available cash, pending settlement, share count, and total account value.
    """
    acct = load_account()
    last_price = yf.Ticker(TICKER).history(period="1d")['Close'].iloc[-1]
    pending_settlement = 0.0

    if os.path.exists(SETTLEMENT_FILE):
        df = pd.read_csv(SETTLEMENT_FILE, parse_dates=["date"])
        pending_settlement = df["amount"].sum()

    total_value = acct['cash'] + acct['shares'] * last_price + pending_settlement

    embed = discord.Embed(
        title="📊 Account Summary",
        color=0x3498db
    )
    embed.add_field(name="Available Cash", value=f"${acct['cash']:.2f}", inline=False)
    embed.add_field(name="Pending Settlement", value=f"${pending_settlement:.2f}", inline=False)
    embed.add_field(name="Shares", value=f"{acct['shares']:.4f}", inline=False)
    embed.add_field(name="Total Account Value", value=f"${total_value:.2f}", inline=False)
    embed.timestamp = ctx.message.created_at

    await ctx.send(embed=embed)

@bot.command()
async def trade(ctx):
    """
    Fetch latest indicators, run ML model, and execute buy/sell if signal threshold met.
    """
    try:
        embed_checking = discord.Embed(
            title="🔍 Trade Check",
            description="Checking for signal...",
            color=0xffc107
        )
        await ctx.send(embed=embed_checking)

        df = fetch_data()
        model, features = await load_model()
        signal, confidence = generate_signal(model, df.iloc[-1], features)
        price = yf.Ticker(TICKER).fast_info['last_price']
        action, shares, price, value, reason = execute_trade(signal, price, confidence)

        if action:
            embed_result = discord.Embed(
                title="✅ Trade Executed",
                color=0x2ecc71
            )
            embed_result.add_field(name="Action", value=action, inline=True)
            embed_result.add_field(name="Shares", value=f"{shares:.4f}", inline=True)
            embed_result.add_field(name="Price", value=f"${price:.2f}", inline=True)
            embed_result.add_field(name="Confidence", value=f"{confidence:.2%}", inline=False)
            embed_result.add_field(name="Reason", value=reason, inline=False)
            embed_result.add_field(name="Account Value", value=f"${value:.2f}", inline=False)
        else:
            embed_result = discord.Embed(
                title="❌ No Trade Executed",
                description=reason,
                color=0xe74c3c
            )

        embed_result.timestamp = ctx.message.created_at
        await ctx.send(embed=embed_result)

    except Exception as e:
        error_msg = f"[ERROR] Exception in !trade: {e}"
        await log_to_channel(error_msg)
        embed_error = discord.Embed(
            title="❌ Trade Error",
            description=error_msg,
            color=0xff0000
        )
        await ctx.send(embed=embed_error)

@bot.command()
async def graph(ctx):
    """
    Plot and send the equity curve (account value) over time as a PNG.
    """
    acct = load_account()
    trades = pd.read_csv(TRADE_LOG)
    
    if trades.empty:
        embed_empty = discord.Embed(
            title="📉 Trade Graph",
            description="No trades to show.",
            color=0xe74c3c
        )
        await ctx.send(embed=embed_empty)
        return

    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    trades['cash_value'] = trades.apply(
        lambda row: row['shares'] * row['price'] if row['action'] == 'SELL' else -row['shares'] * row['price'],
        axis=1
    )
    trades['equity'] = trades['cash_value'].cumsum() + START_BALANCE

    plt.figure(figsize=(10, 5))
    plt.plot(trades['timestamp'], trades['equity'], label='Account Value')
    plt.title('Account Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('USD')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    file = discord.File(fp=buffer, filename='account_value.png')

    embed = discord.Embed(
        title="📈 Account Value Graph",
        description="Here’s your account value progression over time.",
        color=0x1abc9c
    )
    embed.set_image(url="attachment://account_value.png")
    embed.timestamp = ctx.message.created_at

    await ctx.send(embed=embed, file=file)

@bot.command()
async def strat(ctx):
    """
    Describe how the bot trains, when it trades, and which indicators it uses.
    """
    embed = discord.Embed(
        title="📊 Trading Strategy Overview",
        description=(
            "This bot retrains a Gradient Boosting model every day at 3:00 PM CST on the last 2 years of hourly NVDA data, "
            "using a binary target: price up >0.3% over the next 3 hours.\n"
            "During market hours (M–F, 8:30 AM–3:00 PM CST), it checks hourly signals and executes full-position buys or full-position sells."
        ),
        color=0x8e44ad
    )
    embed.add_field(
        name="🔄 Daily Retrain",
        value="- At 3 PM CST, fetch last 2 yrs hourly data\n"
              "- Compute indicators & label 3-hr future returns\n"
              "- Fit GradientBoostingClassifier\n"
              "- Save model & feature list",
        inline=False
    )
    embed.add_field(
        name="⏱️ Trading Logic",
        value="- Hourly: predict probability of price rise >0.3%\n"
              "- Buy if p>60% with all available cash\n"
              "- Sell entire position if p≤60%\n"
              "- Settle sales next morning",
        inline=False
    )
    embed.add_field(
        name="📈 Indicators",
        value="SMA Ratio, RSI, ATR, OBV, Bollinger Width, Stochastic K/D, Donchian Mid & Width",
        inline=False
    )
    embed.add_field(
        name="📌 Commands",
        value="`!account`, `!trade`, `!graph`, `!memory`, `!strat`",
        inline=False
    )
    embed.timestamp = ctx.message.created_at
    await ctx.send(embed=embed)

@tasks.loop(minutes=1)
async def scheduled_trade():
    now = datetime.datetime.now(TIMEZONE)
    holidays = load_holidays()

    channel = bot.get_channel(CHANNEL_ID)

    if now.weekday() >= 5:
        return

    if now.date() in holidays:
        embed_holiday = discord.Embed(
            title="⛔ Market Closed",
            description=f"No trading today - **{now.strftime('%A, %B %d, %Y')}** is a market holiday.",
            color=0xe74c3c
        )
        embed_holiday.timestamp = now
        await channel.send(embed=embed_holiday)
        return

    market_open = now.replace(hour=8, minute=30)
    market_close = now.replace(hour=15, minute=0)

    if now < market_open or now > market_close:
        return

    if now.minute == 31:
        embed_check = discord.Embed(
            title="🕐 Hourly Trade Check",
            description=f"Checking for trade signal at **{now.strftime('%I:%M %p CST')}**.",
            color=0x3498db
        )
        embed_check.timestamp = now
        await channel.send(embed=embed_check)

        await run_trade(channel)

async def run_trade(channel):
    df = fetch_data()
    model, features = await load_model()
    signal, confidence = generate_signal(model, df.iloc[-1], features)
    price = yf.Ticker(TICKER).fast_info['last_price']
    action, shares, price, value, reason = execute_trade(signal, price, confidence)

    now = datetime.datetime.now(TIMEZONE)

    if action:
        embed = discord.Embed(
            title="✅ Trade Executed",
            color=0x2ecc71
        )
        embed.add_field(name="Action", value=action, inline=True)
        embed.add_field(name="Shares", value=f"{shares:.4f}", inline=True)
        embed.add_field(name="Price", value=f"${price:.2f}", inline=True)
        embed.add_field(name="Confidence", value=f"{confidence:.2%}", inline=False)
        embed.add_field(name="Reason", value=reason, inline=False)
        embed.add_field(name="Account Value", value=f"${value:.2f}", inline=False)
    else:
        embed = discord.Embed(
            title="❌ No Trade Executed",
            description=reason,
            color=0xe74c3c
        )

    embed.timestamp = now

    await channel.send(embed=embed)

@bot.command()
async def test(ctx):
    """
    Compare last model prediction against historical indicator stats.
    """
    try:
        df = pd.read_csv(DATASET_FILE)

        for col in ['Close', 'SMA_10', 'SMA_50', 'RSI', 'ATR', 'OBV', 'BB_Width', 'Stochastic_K', 'Stochastic_D']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        if 'Target' not in df.columns:
            df['FutureReturn'] = df['Close'].shift(-3) / df['Close'] - 1
            df['Target'] = np.where(df['FutureReturn'] > 0.001, 1, 0)
            df.drop(columns=['FutureReturn'], inplace=True)
            df.dropna(inplace=True)

        model, features = await load_model()
        latest_df = fetch_data()
        latest_row = latest_df.iloc[-1]
        values = latest_row[features].astype(float).values.reshape(1, -1)
        proba = model.predict_proba(values)[0, 1]
        prediction = "UP 📈" if proba > 0.6 else "DOWN 📉"
        confidence = proba if proba > 0.6 else 1 - proba

        expected_up_means = df[df['Target'] == 1][features].mean()
        expected_up_std = df[df['Target'] == 1][features].std()

        indicators = []
        for feat in features:
            current = latest_row[feat].item()
            expected = expected_up_means[feat]
            std = expected_up_std[feat]
            deviation = 0 if std == 0 or np.isnan(std) else abs(current - expected) / std
            emoji, arrow = (
                ("✅", "⬆️") if deviation < 0.5 else
                ("⚠️", "↔") if deviation < 1.0 else
                ("❌", "⬇️")
            )
            indicators.append(f"{emoji} **{feat}**: {current:.4f} | expected: {expected:.4f} {arrow}")

        prediction_vals = (
            f"Class 0 (↓): {1 - proba:.2%}\n"
            f"Class 1 (↑): {proba:.2%}"
        )

        embed = discord.Embed(
            title="📊 Model Test Summary",
            color=0x1f77b4
        )
        embed.add_field(
            name="🔎 Current vs Expected Indicators",
            value="\n".join(indicators),
            inline=False
        )
        embed.add_field(
            name="📈 Model Prediction Probabilities",
            value=prediction_vals,
            inline=False
        )
        embed.add_field(
            name="🧠 Final Signal",
            value=f"**{prediction}** (Confidence: {confidence:.2%})",
            inline=False
        )
        embed.timestamp = ctx.message.created_at

        await ctx.send(embed=embed)

    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Error",
            description=f"Exception in `!test`: `{e}`",
            color=0xff0000
        )
        await ctx.send(embed=error_embed)

@bot.command()
async def why(ctx):
    """
    Generate SHAP breakdown explaining last model prediction.
    """
    try:
        model, features = await load_model()
        df = pd.read_csv(DATASET_FILE)
        df = df[features]

        if df.empty:
            raise ValueError("The dataset is empty. The model has not seen any data yet.")

        latest_row = df.tail(1)
        explainer = shap.Explainer(model, df)
        shap_values = explainer(latest_row)
        shap_array = shap_values.values[0]

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(shap_values[0], show=False, ax=ax)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        file = discord.File(fp=buffer, filename='shap_explain.png')

        indicator_descriptions = {
            "SMA_Ratio": "Ratio of short-term (10-period) to long-term (50-period) moving average. Higher means bullish trend.",
            "RSI": "Relative Strength Index (0-100). Over 70 is overbought, under 30 is oversold.",
            "ATR": "Average True Range, a volatility measure. Higher = more price movement.",
            "OBV": "On-Balance Volume. Captures volume trend confirmation with price.",
            "BB_Width": "Bollinger Band width. Expands with volatility and potential breakouts.",
            "Stochastic_K": "%K value of stochastic oscillator. Measures close relative to recent highs/lows.",
            "Stochastic_D": "%D value (smoothed %K). Signal line for stochastic crossovers.",
            "Donchian_Mid": "Midpoint of Donchian Channel (20-period high/low). Shows breakout trends.",
            "Donchian_Width": "Width of Donchian Channel. Measures recent range volatility."
        }

        sorted_features = sorted(zip(features, shap_array), key=lambda x: abs(x[1]), reverse=True)
        full_explanation = []
        for feat, val in sorted_features:
            effect = "boosted" if val > 0 else "reduced"
            impact = f"`{val:+.4f}`"
            desc = indicator_descriptions.get(feat, "")
            full_explanation.append(f"**{feat}** ({desc})\n→ {effect} model confidence by {impact}\n")

        embed = discord.Embed(
            title="🔍 Full SHAP Indicator Breakdown",
            description="Feature-level explanation for the model's last decision.",
            color=0x9b59b6
        )
        embed.set_image(url="attachment://shap_explain.png")
        embed.add_field(name="📘 Feature Insights", value="\n".join(full_explanation[:5]), inline=False)
        embed.timestamp = ctx.message.created_at

        await ctx.send(embed=embed, file=file)

    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Error",
            description=f"Exception in `!why`: `{e}`",
            color=0xff0000
        )
        await ctx.send(embed=error_embed)

@bot.command()
async def profit(ctx):
    """
    Show current profit/loss relative to the starting balance, $100.
    """
    acct = load_account()
    last_price = yf.Ticker(TICKER).history(period="1d")['Close'].iloc[-1]

    pending_settlement = 0.0
    if os.path.exists(SETTLEMENT_FILE):
        df = pd.read_csv(SETTLEMENT_FILE, parse_dates=["date"])
        pending_settlement = df["amount"].sum()

    total_value = acct['cash'] + acct['shares'] * last_price + pending_settlement

    embed = discord.Embed(
        title="💰 Profit/Loss",
        color=0x6b1a7d
    )
    if (total_value-100 > 0):
        embed.add_field(
            name="📈 Profit",
            value=f"Currently up ${total_value-100}",
            inline=False
        )
    elif (total_value-100 < 0):
        embed.add_field(
            name="📉 Loss",
            value=f"Currently down ${total_value-100}",
            inline=False
        )
    else:
        embed.add_field(
            name="📊 Break-even",
            value=f"Currently at ${total_value-100}",
            inline=False
        )
    embed.timestamp = ctx.message.created_at
    await ctx.send(embed=embed)

@bot.command()
async def system(ctx):
    """
    Report CPU, memory, disk, load average, process count, network I/O and uptime.
    """
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    current_rss = mem.rss / 1024**2
    peak_rss    = getattr(mem, 'peak_wset', mem.rss) / 1024**2
    cpu = proc.cpu_percent(interval=0.1)
    load1, load5, load15 = psutil.getloadavg()
    threads = proc.num_threads()
    total_procs = len(psutil.pids())
    du = shutil.disk_usage("/")
    disk_used = (du.used / du.total) * 100
    net = psutil.net_io_counters()
    sent_mb = net.bytes_sent / 1024**2
    recv_mb = net.bytes_recv / 1024**2
    uptime = datetime.datetime.now(TIMEZONE) - datetime.datetime.fromtimestamp(proc.create_time(), TIMEZONE)

    embed = discord.Embed(title="🖥️ System Usage", color=0x95a5a6)
    embed.add_field(name="CPU Usage",      value=f"{cpu:.1f} %", inline=True)
    embed.add_field(name="Load Avg (1/5/15m)", value=f"{load1:.2f}/{load5:.2f}/{load15:.2f}", inline=True)
    embed.add_field(name="Current RSS",    value=f"{current_rss:.1f} MB", inline=True)
    embed.add_field(name="Peak RSS",       value=f"{peak_rss:.1f} MB", inline=True)
    embed.add_field(name="Threads",        value=str(threads), inline=True)
    embed.add_field(name="Processes",      value=str(total_procs), inline=True)
    embed.add_field(name="Disk Used",      value=f"{disk_used:.1f} %", inline=True)
    embed.add_field(name="Net Sent/Recv",  value=f"{sent_mb:.1f} MB / {recv_mb:.1f} MB", inline=False)
    embed.add_field(name="Uptime",         value=str(uptime).split('.')[0], inline=False)
    await ctx.send(embed=embed)

@bot.event
async def on_ready():
    init_account()
    await log_to_channel("[INFO] Bot is initializing model...")

    if not os.path.exists(MODEL_FILE):
        await log_to_channel("[INFO] Training initial model...")
        model, features = await train_initial_model()
    else:
        model, features = await load_model()
        await log_to_channel("[INFO] Model found. Skipping training.")

    importances = model.feature_importances_
    ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    importance_summary = "\n".join([f"**{feat}**: {imp:.3f}" for feat, imp in ranked])

    acct = load_account()
    last_price = yf.Ticker(TICKER).history(period="1d")['Close'].iloc[-1]
    pending_settlement = 0.0
    if os.path.exists(SETTLEMENT_FILE):
        df = pd.read_csv(SETTLEMENT_FILE, parse_dates=["date"])
        pending_settlement = df["amount"].sum()
    total_value = acct['cash'] + acct['shares'] * last_price + pending_settlement

    trades = pd.read_csv(TRADE_LOG)
    trade_count = len(trades) if not trades.empty else 0
    days_running = (datetime.datetime.now(TIMEZONE) - START_TIME).days

    channel = bot.get_channel(CHANNEL_ID)
    embed = discord.Embed(
        title=f"✅ {TICKER} Bot Online",
        description="The ML trading bot is now live and operational.",
        color=0x2ecc71
    )
    embed.add_field(name="Model Status", value="✅ Loaded" if os.path.exists(MODEL_FILE) else "🧠 Trained", inline=True)
    embed.add_field(name="Tasks", value="Scheduled tasks started successfully.", inline=True)
    embed.add_field(name="📊 Feature Importance", value=importance_summary, inline=False)
    embed.add_field(
        name="📈 Bot Stats",
        value=f"- Days Running: **{days_running}**\n"
              f"- Trades Executed: **{trade_count}**\n"
              f"- Account Value: **${total_value:.2f}**",
        inline=False
    )
    embed.timestamp = datetime.datetime.now(TIMEZONE)

    await channel.send(embed=embed)

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
