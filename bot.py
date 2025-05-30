import requests
import numpy as np
import pandas as pd
import logging
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# API Keys
TWELVE_API_KEY = '14ee1e5c333945c190f19c097138bdd5'
TELEGRAM_TOKEN = '8030718150:AAFp5QuwaC-103ruvB5TsBMGY5MwMvkq-5g'

# Constants
SYMBOL = 'USD/JPY'
INTERVAL = '1min'
PERIOD = 14

# Fetch Data
def fetch_data(symbol=SYMBOL, interval=INTERVAL, outputsize=100):
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_API_KEY}'
    response = requests.get(url)
    data = response.json()

    if 'values' not in data:
        raise Exception(f"Error fetching data: {data.get('message', 'Unknown error')}")

    df = pd.DataFrame(data['values'])
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# Technical Indicators
def calculate_rsi(series, period=PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + std_dev * std, sma - std_dev * std

def calculate_adx(high, low, close, period=PERIOD):
    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    return dx.rolling(window=period).mean()

def calculate_stoch_rsi(rsi_series, period=PERIOD):
    min_rsi = rsi_series.rolling(period).min()
    max_rsi = rsi_series.rolling(period).max()
    return 100 * (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-10)

def calculate_vwap(df):
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

def calculate_atr(high, low, close, period=PERIOD):
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# Analyze Market Signals
def analyze_signals(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    rsi = calculate_rsi(close)
    macd, signal, hist = calculate_macd(close)
    upper_bb, lower_bb = calculate_bollinger_bands(close)
    adx = calculate_adx(high, low, close)
    stoch_rsi = calculate_stoch_rsi(rsi)
    vwap = calculate_vwap(df)
    atr = calculate_atr(high, low, close)

    latest = -1
    signals = []
    confidence = 0

    # RSI
    if rsi.iloc[latest] < 30:
        signals.append("📉 RSI: Oversold (Bullish)")
        confidence += 15
    elif rsi.iloc[latest] > 70:
        signals.append("📈 RSI: Overbought (Bearish)")
        confidence += 15

    # MACD
    if macd.iloc[latest] > signal.iloc[latest]:
        signals.append("📈 MACD Bullish Crossover")
        confidence += 15
    else:
        signals.append("📉 MACD Bearish Crossover")
        confidence += 15

    # Bollinger Bands
    if close.iloc[latest] < lower_bb.iloc[latest]:
        signals.append("📉 Price below lower Bollinger Band (Bullish)")
        confidence += 10
    elif close.iloc[latest] > upper_bb.iloc[latest]:
        signals.append("📈 Price above upper Bollinger Band (Bearish)")
        confidence += 10

    # ADX
    if adx.iloc[latest] > 25:
        signals.append("📊 ADX: Strong Trend")
        confidence += 10
    else:
        signals.append("📉 ADX: Weak/No Trend")

    # Stochastic RSI
    if stoch_rsi.iloc[latest] < 20:
        signals.append("📉 Stoch RSI: Oversold (Bullish)")
        confidence += 10
    elif stoch_rsi.iloc[latest] > 80:
        signals.append("📈 Stoch RSI: Overbought (Bearish)")
        confidence += 10

    # VWAP
    if close.iloc[latest] > vwap.iloc[latest]:
        signals.append("📈 Price above VWAP (Bullish)")
        confidence += 10
    else:
        signals.append("📉 Price below VWAP (Bearish)")

    # ATR
    signals.append(f"🔄 ATR (Volatility): {atr.iloc[latest]:.5f}")

    confidence = min(confidence, 100)
    bullish = sum("Bullish" in s for s in signals)
    bearish = sum("Bearish" in s for s in signals)

    if bullish > bearish:
        recommendation = "✅ Strong BUY signal"
    elif bearish > bullish:
        recommendation = "⚠️ Strong SELL signal"
    else:
        recommendation = "⚖️ Neutral / Wait"

    return signals, confidence, recommendation

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Welcome to the Forex Trade Bot!\nUse /trade to get the latest USD/JPY signals.")

async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("⏳ Fetching market data and analyzing indicators...")
        df = fetch_data()
        signals, confidence, recommendation = analyze_signals(df)

        message = "📊 USD/JPY 1-Minute Analysis:\n\n"
        for s in signals:
            message += f"• {s}\n"
        message += f"\n🔒 Confidence Level: {confidence}%\n📌 Recommendation: {recommendation}"

        await update.message.reply_text(message)

    except Exception as e:
        logging.error("Error during trade analysis", exc_info=True)
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")

# Main Bot App
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("trade", trade))
    logging.info("✅ Bot is running...")
    await app.run_polling()

# Run Script
if __name__ == "__main__":
    asyncio.run(main())
