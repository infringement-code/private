import discord
import datetime
import json
import aiohttp
import mplfinance as mpf
import io
import asyncio
from discord.ext import commands, tasks
import ccxt
import pandas as pd
import pandas_ta as ta
import os
from dotenv import load_dotenv
import sqlite3

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
GROK_API_KEY = os.getenv("GROK_API_KEY")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

exchange = ccxt.okx({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# ==================== SETTINGS ====================
MAX_COINS = 100
MIN_24H_VOLUME_USDT = 100_000

STABLECOINS = {"USDC/USDT", "USD1/USDT", "USDT/USDT", "BUSD/USDT", "TUSD/USDT", "FDUSD/USDT", "USDD/USDT"}

last_signal_time = {}
SCALP_COOLDOWN = 1800
SWING_COOLDOWN = 7200
SPOT_COOLDOWN = 21600

DYNAMIC_WATCHLIST = []
# ====================================================

# ==================== TUNABLE PRE-FILTER ====================
MIN_VOLUME_RATIO = 0.6   # Lower = more signals (0.5 = very loose)

SCALP_RSI_LONG_MIN  = 22
SCALP_RSI_LONG_MAX  = 48
SCALP_RSI_SHORT_MIN = 52
SCALP_RSI_SHORT_MAX = 78

SWING_RSI_LONG_MIN  = 25
SWING_RSI_LONG_MAX  = 55
SWING_RSI_SHORT_MIN = 45
SWING_RSI_SHORT_MAX = 75
# ========================================================
# ==================== LOGGING TOGGLES ====================
LOG_PREFILTER = False     # Set to False to reduce noise
LOG_SCALP = False         # Set to False to reduce noise
LOG_GROK = True          # Keep this True when debugging API
# ========================================================
@bot.event
async def on_ready():
    print(f"✅ {bot.user} is online!")
    init_db()
    print("💾 Signal database ready")
    print("🔄 Fetching initial dynamic watchlist...")
    await refresh_watchlist()
    # No automatic test signal on startup anymore
    signal_loop.start()
    refresh_watchlist.start()
    check_open_signals.start()

@tasks.loop(hours=1)
async def refresh_watchlist():
    global DYNAMIC_WATCHLIST
    try:
        print("🔄 Refreshing dynamic market scanner...")
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, data in tickers.items():
            if (symbol.endswith("/USDT") and
                data.get("quoteVolume", 0) >= MIN_24H_VOLUME_USDT and
                symbol not in STABLECOINS):
                usdt_pairs.append((symbol, data.get("quoteVolume", 0)))
        
        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        DYNAMIC_WATCHLIST = [pair[0] for pair in usdt_pairs[:MAX_COINS]]
        
        print(f"✅ Dynamic scanner loaded {len(DYNAMIC_WATCHLIST)} coins (top volume, no stables)")
        if DYNAMIC_WATCHLIST:
            print(f"   Top 5: {', '.join(DYNAMIC_WATCHLIST[:5])}")
    except Exception as e:
        print(f"⚠️ Watchlist refresh failed: {e}")
        if not DYNAMIC_WATCHLIST:
            DYNAMIC_WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

@tasks.loop(minutes=5)
async def signal_loop():
    if not DYNAMIC_WATCHLIST:
        print("⚠️ No coins in dynamic watchlist yet...")
        return
    print(f"🔄 Running AI scan on {len(DYNAMIC_WATCHLIST)} live coins...")

    for symbol in DYNAMIC_WATCHLIST:
        current_symbol = symbol
        try:
            print(f"   [SCALP] Checking {current_symbol}...")
            df = await fetch_ohlcv(symbol, '5m', limit=200)
            signal = await generate_ai_signal(df, symbol, "SCALP", "5m", "scalp-signals")
            if signal:
                await send_signal_to_channel(signal, "scalp-signals")
            await asyncio.sleep(0.25)
        except Exception as e:
            print(f"⚠️ Scalp error on {current_symbol}: {e}")

    # Swing every 15 min
    if datetime.datetime.now().minute % 15 == 0:
        for symbol in DYNAMIC_WATCHLIST:
            current_symbol = symbol
            try:
                print(f"   [SWING] Checking {current_symbol}...")
                df = await fetch_ohlcv(symbol, '1h', limit=200)
                signal = await generate_ai_signal(df, symbol, "SWING", "1h", "swing-signals")
                if signal:
                    await send_signal_to_channel(signal, "swing-signals")
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"⚠️ Swing error on {current_symbol}: {e}")

    # Spot every 30 min
    if datetime.datetime.now().minute % 30 == 0:
        for symbol in DYNAMIC_WATCHLIST:
            current_symbol = symbol
            try:
                print(f"   [SPOT] Checking {current_symbol}...")
                df = await fetch_ohlcv(symbol, '4h', limit=200)
                signal = await generate_ai_signal(df, symbol, "SPOT", "4h", "spot-signals")
                if signal:
                    await send_signal_to_channel(signal, "spot-signals")
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"⚠️ Spot error on {current_symbol}: {e}")

async def fetch_ohlcv(symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def passes_quantitative_filter(df, symbol: str, signal_type: str) -> bool:
    if len(df) < 50:
        print(f"   [Pre-filter] {symbol} {signal_type} → REJECTED (not enough data)")
        return False

    close = df['close'].iloc[-1]
    rsi = ta.rsi(df['close'], length=14).iloc[-1]
    ema9 = ta.ema(df['close'], length=9).iloc[-1]
    volume = df['volume'].iloc[-1]
    volume_sma = ta.sma(df['volume'], length=20).iloc[-1]
    volume_ratio = volume / volume_sma

    print(f"   [Pre-filter] {symbol} {signal_type} | Price=${close:.4f} | RSI={rsi:.1f} | Vol={volume_ratio:.2f}x | vs EMA9={'above' if close > ema9 else 'below'}")

    if volume_ratio < MIN_VOLUME_RATIO:
        print(f"   [Pre-filter] {symbol} {signal_type} → REJECTED (volume too low)")
        return False

    if signal_type == "SCALP":
        if close > ema9 and SCALP_RSI_LONG_MIN < rsi < SCALP_RSI_LONG_MAX:
            print(f"   [Pre-filter] {symbol} {signal_type} → **PASSED** (SCALP LONG)")
            return True
        if close < ema9 and SCALP_RSI_SHORT_MIN < rsi < SCALP_RSI_SHORT_MAX:
            print(f"   [Pre-filter] {symbol} {signal_type} → **PASSED** (SCALP SHORT)")
            return True

    elif signal_type in ["SWING", "SPOT"]:
        if close > ema9 and SWING_RSI_LONG_MIN < rsi < SWING_RSI_LONG_MAX:
            print(f"   [Pre-filter] {symbol} {signal_type} → **PASSED** (SWING/SPOT LONG)")
            return True
        if close < ema9 and SWING_RSI_SHORT_MIN < rsi < SWING_RSI_SHORT_MAX:
            print(f"   [Pre-filter] {symbol} {signal_type} → **PASSED** (SWING/SPOT SHORT)")
            return True

    print(f"   [Pre-filter] {symbol} {signal_type} → REJECTED (no strong setup)")
    return False

async def analyze_with_grok(df, symbol, timeframe):
    if LOG_GROK:
        print(f"   [Grok API] Starting call for {symbol} {timeframe}...")

    try:
        recent = df.tail(20).copy()
        recent['rsi'] = ta.rsi(recent['close'], length=14)
        recent['macd'] = ta.macd(recent['close'])['MACD_12_26_9']
        recent['ema9'] = ta.ema(recent['close'], length=9)
        recent['ema21'] = ta.ema(recent['close'], length=21)
        recent['atr'] = ta.atr(recent['high'], recent['low'], recent['close'], length=14)
        recent['volume_sma'] = ta.sma(recent['volume'], length=20)

        data_str = recent[['close', 'rsi', 'macd', 'ema9', 'ema21', 'atr', 'volume', 'volume_sma']].round(4).tail(10).to_string()

        prompt = f"""You are a professional crypto trader specializing in volatile meme and alt coins.

Current {timeframe} chart for {symbol}:
{data_str}

Current price: ${recent['close'].iloc[-1]:.4f}
RSI: {recent['rsi'].iloc[-1]:.1f}
Volume vs avg: {recent['volume'].iloc[-1] / recent['volume_sma'].iloc[-1]:.2f}x

Be strict. Only signal if you see a high-probability setup.

Respond **ONLY** with valid JSON:
{{
  "action": "LONG" or "SHORT" or "HOLD",
  "confidence": 0-100,
  "stop_loss_pct": -2.5 to -6.0,
  "reason": "one short professional sentence"
}}

Only return signal if confidence >= 75. Otherwise use "HOLD".
"""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
                json={"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "temperature": 0.15, "max_tokens": 300}
            ) as resp:
                
                if LOG_GROK:
                    print(f"   [Grok API] Status code for {symbol} {timeframe}: {resp.status}")
                    raw_text = await resp.text()
                    print(f"   [Grok API] Raw response for {symbol} {timeframe}:")
                    print(raw_text[:1000])  # first 1000 chars only

                if resp.status != 200:
                    print(f"⚠️ Grok API returned non-200 status ({resp.status}) for {symbol}")
                    return {"action": "HOLD", "confidence": 0, "stop_loss_pct": 0, "reason": "API error"}

                result = await resp.json()

                if not result or 'choices' not in result or not result.get('choices'):
                    print(f"⚠️ Grok API Bad Response for {symbol} {timeframe}")
                    return {"action": "HOLD", "confidence": 0, "stop_loss_pct": 0, "reason": "API error"}

                try:
                    usage = result.get('usage', {})
                    input_t = usage.get('input_tokens', 0)
                    output_t = usage.get('output_tokens', 0)
                    cost = (input_t * 3.00 + output_t * 15.00) / 1_000_000
                    if LOG_GROK:
                        print(f"🔥 Grok {symbol} {timeframe} | {input_t}+{output_t} tokens | Cost: ${cost:.5f}")
                except:
                    pass

                content = result['choices'][0]['message']['content']
                return json.loads(content)

    except Exception as e:
        print(f"⚠️ Grok API exception for {symbol} {timeframe}: {e}")
        return {"action": "HOLD", "confidence": 0, "stop_loss_pct": 0, "reason": "API error"}


async def generate_ai_signal(df, symbol, signal_type, tf_str, channel):
    print(f"   [Pre-filter] Checking {symbol} {signal_type}...")
    if not passes_quantitative_filter(df, symbol, signal_type):
        return None

    key = f"{symbol}-{signal_type}"
    cooldown = SCALP_COOLDOWN if signal_type == "SCALP" else SWING_COOLDOWN if signal_type == "SWING" else SPOT_COOLDOWN
    if key in last_signal_time and (datetime.datetime.utcnow() - last_signal_time[key]).total_seconds() < cooldown:
        print(f"   [Cooldown] Skipping {symbol} {signal_type}")
        return None

    grok = await analyze_with_grok(df, symbol, tf_str)
    if grok["action"] == "HOLD" or grok["confidence"] < 75:
        print(f"   [Grok] HOLD on {symbol} {signal_type}")
        return None

    last_signal_time[key] = datetime.datetime.utcnow()
    print(f"   [Grok] SIGNAL ACCEPTED for {symbol} {signal_type} (Conf: {grok['confidence']}%)")

    close = df['close'].iloc[-1]
    entry = round(close, 4 if close < 10 else 2)
    tp_pcts = [4, 8, 15, 25, 40, 60] if signal_type == "SCALP" else [5, 12, 20, 30, 50, 80]
    direction = 1 if grok["action"] == "LONG" else -1
    tps = [round(entry * (1 + direction * (p / 100)), 4 if entry < 10 else 2) for p in tp_pcts]

    return {
        "type": signal_type,
        "brand": "aMe Signals",
        "pair": symbol,
        "action": grok["action"],
        "entry": entry,
        "timeframe": tf_str,
        "stop_loss": round(entry * (1 + grok["stop_loss_pct"]/100), 4 if entry < 10 else 2),
        "tps": tps,
        "tp_pcts": tp_pcts,
        "confidence": grok["confidence"],
        "strategy": f"Premium aMe {signal_type} Signal",
        "utc_time": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": grok["reason"]
    }

# ==================== SIGNAL SENDING ====================
async def send_signal_to_channel(signal, channel_name):
    color = 0x00ff88 if signal["action"] == "LONG" else 0xff3333
    embed = discord.Embed(
        title=f"🚀 💎 {signal['brand']} {signal['type']} SIGNAL 💎 🚀",
        color=color,
        timestamp=datetime.datetime.now(datetime.UTC)
    )
    embed.set_author(name="aMe Signals APP", icon_url=bot.user.display_avatar.url)
    chart_file = await generate_chart(signal["pair"], signal["timeframe"])

    desc = f"""
📊 **Symbol:** {signal['pair'].replace('/', '')}

────────────────────────────

💰 **Entry Price:** ${signal['entry']}
⏰ **Timeframe:** {signal['timeframe']}
🛡️ **Stop Loss:** ${signal['stop_loss']} 

────────────────────────────

🎯 **PROFIT TARGETS:**
"""
    tp_emojis = ["🥇", "🥈", "🥉", "🏆", "⭐", "💎"]
    for i, (price, pct) in enumerate(zip(signal["tps"], signal["tp_pcts"])):
        desc += f"{tp_emojis[i]} **TP{i+1}:** ${price} (+{pct}%)\n"

    desc += f"""
────────────────────────────

💎 **Strategy:** {signal['strategy']}
🔥 **Confidence:** {signal['confidence']}% 
📝 **Reason:** {signal['reason']}

────────────────────────────

⏰ **UTC Time:** {signal['utc_time']}
"""
    embed.description = desc.strip()
    embed.set_footer(text="Not financial advice • DYOR • High risk • Trade responsibly")

    channel = discord.utils.get(bot.get_all_channels(), name=channel_name)
    if channel:
        if chart_file:
            embed.set_image(url=f"attachment://{chart_file.filename}")
            await channel.send(embed=embed, file=chart_file)
            save_signal(signal)
            print(f"✅ {signal['type']} {signal['action']} on {signal['pair']} | Chart embedded")
        else:
            await channel.send(embed=embed)
            save_signal(signal)
            print(f"✅ {signal['type']} {signal['action']} on {signal['pair']} (no chart)")
    else:
        print(f"⚠️ Channel #{channel_name} not found!")

async def send_test_signal_to_channel(signal, channel_name):
    color = 0x00ff88 if signal["action"] == "LONG" else 0xff3333
    embed = discord.Embed(
        title=f"🚀 💎 {signal['brand']} {signal['type']} SIGNAL 💎 🚀 (TEST)",
        color=color,
        timestamp=datetime.datetime.now(datetime.UTC)
    )
    embed.set_author(name="aMe Signals APP", icon_url=bot.user.display_avatar.url)
    chart_file = await generate_chart(signal["pair"], signal["timeframe"])

    desc = f"""
📊 **Symbol:** {signal['pair'].replace('/', '')}

────────────────────────────

💰 **Entry Price:** ${signal['entry']}
⏰ **Timeframe:** {signal['timeframe']}
🛡️ **Stop Loss:** ${signal['stop_loss']} 

────────────────────────────

🎯 **PROFIT TARGETS:**
"""
    tp_emojis = ["🥇", "🥈", "🥉", "🏆", "⭐", "💎"]
    for i, (price, pct) in enumerate(zip(signal["tps"], signal["tp_pcts"])):
        desc += f"{tp_emojis[i]} **TP{i+1}:** ${price} (+{pct}%)\n"

    desc += f"""
────────────────────────────

💎 **Strategy:** {signal['strategy']}
🔥 **Confidence:** {signal['confidence']}% 
📝 **Reason:** {signal['reason']}

────────────────────────────

⏰ **UTC Time:** {signal['utc_time']}
"""
    embed.description = desc.strip()
    embed.set_footer(text="TEST SIGNAL • Not counted in performance • Not financial advice")

    channel = discord.utils.get(bot.get_all_channels(), name=channel_name)
    if channel:
        if chart_file:
            embed.set_image(url=f"attachment://{chart_file.filename}")
            await channel.send(embed=embed, file=chart_file)
        else:
            await channel.send(embed=embed)
        print(f"✅ TEST signal sent to #{channel_name} (not saved to DB)")
    else:
        print(f"⚠️ Channel #{channel_name} not found!")

async def generate_chart(pair, timeframe):
    try:
        df = await fetch_ohlcv(pair, timeframe, limit=120)
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df['EMA9'] = ta.ema(df['close'], length=9)
        df['EMA21'] = ta.ema(df['close'], length=21)

        mc = mpf.make_marketcolors(up='#00ff88', down='#ff3333', wick={'up':'#00ff88','down':'#ff3333'}, volume={'up':'#00ff88','down':'#ff3333'}, inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2a2a', facecolor='#0e0e0e', figcolor='#0e0e0e', rc={'font.size': 10})

        ap = [
            mpf.make_addplot(df['EMA9'], color='#f4a261', width=1.2),
            mpf.make_addplot(df['EMA21'], color='#4fc3f7', width=1.2)
        ]

        buf = io.BytesIO()
        mpf.plot(df, type='candle', style=s, volume=True, addplot=ap,
                 title=f"{pair} {timeframe} • aMe Signals",
                 figsize=(13, 8), savefig=buf, panel_ratios=(3,1), volume_panel=1,
                 tight_layout=True, scale_padding=0.15)
        buf.seek(0)
        return discord.File(buf, filename=f"{pair}_{timeframe}_chart.png")
    except Exception as e:
        print(f"⚠️ Chart generation failed for {pair}: {e}")
        return None

# ==================== DATABASE ====================
def init_db():
    conn = sqlite3.connect('signals.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT,
        signal_type TEXT,
        action TEXT,
        entry REAL,
        stop_loss REAL,
        tps TEXT,
        entry_time TEXT,
        status TEXT DEFAULT 'OPEN',
        exit_price REAL,
        pnl_percent REAL,
        pnl_dollars REAL,
        hit_level TEXT
    )''')
    conn.commit()
    conn.close()

def save_signal(signal):
    conn = sqlite3.connect('signals.db')
    c = conn.cursor()
    tps_json = json.dumps(signal['tps'])
    c.execute('''INSERT INTO signals 
                 (pair, signal_type, action, entry, stop_loss, tps, entry_time)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (signal['pair'], signal['type'], signal['action'],
               signal['entry'], signal['stop_loss'], tps_json, signal['utc_time']))
    conn.commit()
    conn.close()
    print(f"💾 Saved signal for {signal['pair']} to database")

@tasks.loop(minutes=5)
async def check_open_signals():
    conn = sqlite3.connect('signals.db')
    c = conn.cursor()
    c.execute("SELECT * FROM signals WHERE status = 'OPEN'")
    open_signals = c.fetchall()
    conn.close()
    if not open_signals:
        return
    print(f"🔍 Checking {len(open_signals)} open calls...")
    for row in open_signals:
        sid, pair, sig_type, action, entry, sl, tps_json, entry_time, status, _, _, _, _ = row
        try:
            ticker = exchange.fetch_ticker(pair)
            current = ticker['last']
            tps = json.loads(tps_json)
            hit = None
            pnl_pct = 0
            if action == "LONG":
                if current <= sl:
                    hit = "SL"
                    pnl_pct = (sl - entry) / entry * 100
                else:
                    for i, tp in enumerate(tps):
                        if current >= tp:
                            hit = f"TP{i+1}"
                            pnl_pct = (tp - entry) / entry * 100
                            break
            else:
                if current >= sl:
                    hit = "SL"
                    pnl_pct = (entry - sl) / entry * 100
                else:
                    for i, tp in enumerate(tps):
                        if current <= tp:
                            hit = f"TP{i+1}"
                            pnl_pct = (entry - tp) / entry * 100
                            break
            if hit:
                pnl_dollars = round(pnl_pct / 100 * 1000, 2)
                conn = sqlite3.connect('signals.db')
                c = conn.cursor()
                c.execute("""UPDATE signals 
                             SET status=?, exit_price=?, pnl_percent=?, pnl_dollars=?, hit_level=?
                             WHERE id=?""",
                          (hit if hit == "SL" else "CLOSED", current, pnl_pct, pnl_dollars, hit, sid))
                conn.commit()
                conn.close()
                print(f"✅ {pair} hit {hit} → ${pnl_dollars}")
        except:
            pass


@bot.command()
async def testsignal(ctx, signal_type: str = "scalp"):
    await ctx.send(f"🧪 Generating real Grok test {signal_type.upper()} signal for BTC/USDT...")
    await send_test_signal(f"{signal_type}-signals")


async def send_test_signal(channel_name):
    """Forces a real Grok API call for testing"""
    print("🧪 [Test] Starting real Grok API call for BTC/USDT...")
    try:
        df = await fetch_ohlcv('BTC/USDT', '5m', limit=200)
        signal = await generate_ai_signal(df, 'BTC/USDT', "SCALP", "5m", "scalp-signals")
        
        if signal:
            await send_signal_to_channel(signal, channel_name)
            print("✅ [Test] Real Grok signal sent successfully")
        else:
            print("   [Test] Grok returned HOLD - no signal generated")
    except Exception as e:
        print(f"Test signal error: {e}")
        
@bot.command()
async def calls(ctx):
    """Show all currently open calls"""
    conn = sqlite3.connect('signals.db')
    c = conn.cursor()
    c.execute("SELECT * FROM signals WHERE status = 'OPEN' ORDER BY entry_time DESC")
    rows = c.fetchall()
    conn.close()

    if not rows:
        await ctx.send("✅ No open calls right now!")
        return

    embed = discord.Embed(title="📬 OPEN CALLS", color=0x00ffff)
    for row in rows[:10]:  # max 10
        pair, sig_type, action, entry, sl = row[1], row[2], row[3], row[4], row[5]
        embed.add_field(name=f"{pair} {action} ({sig_type})", 
                        value=f"Entry: ${entry}\nSL: ${sl}", inline=False)
    await ctx.send(embed=embed)

@bot.command()
async def performance(ctx):
    """Full P&L report with $1,000 starting capital simulation"""
    conn = sqlite3.connect('signals.db')
    c = conn.cursor()
    c.execute("SELECT * FROM signals")
    rows = c.fetchall()
    conn.close()

    if not rows:
        await ctx.send("No signals yet!")
        return

    total_pnl = 0
    wins = 0
    closed = 0
    open_trades = []

    for row in rows:
        status = row[8]
        pnl_d = row[11] or 0
        total_pnl += pnl_d
        if status != "OPEN":
            closed += 1
            if pnl_d > 0:
                wins += 1
        else:
            open_trades.append(row)

    win_rate = round((wins / closed * 100), 1) if closed else 0

    embed = discord.Embed(title="📊 aMe Signals PERFORMANCE", color=0x00ff88)
    embed.add_field(name="Total P&L", 
                    value=f"${total_pnl:,.2f} (from $1k/trade)", inline=False)
    embed.add_field(name="Closed Trades", value=f"{closed} | Win Rate: {win_rate}%", inline=True)
    embed.add_field(name="Open Trades", value=len(open_trades), inline=True)

    if open_trades:
        embed.add_field(name="🔴 Currently Open", value="Check with `!calls`", inline=False)

    embed.set_footer(text="Simulated with $1,000 notional per trade • Not financial advice")
    await ctx.send(embed=embed)

@bot.command()
async def history(ctx):
    """Shows all closed trades with full P&L details"""
    conn = sqlite3.connect('signals.db')
    c = conn.cursor()
    c.execute("SELECT * FROM signals WHERE status != 'OPEN' ORDER BY entry_time DESC")
    rows = c.fetchall()
    conn.close()

    if not rows:
        await ctx.send("No closed trades yet!")
        return

    embed = discord.Embed(title="📜 aMe Signals HISTORY (Last 15 closed)", color=0x00ff88)
    for row in rows[:15]:
        pair = row[1]
        sig_type = row[2]
        action = row[3]
        entry = row[4]
        hit = row[12]
        pnl = row[11]
        emoji = "🟢" if pnl and pnl > 0 else "🔴"
        embed.add_field(
            name=f"{emoji} {pair} {action} ({sig_type})",
            value=f"Entry: ${entry}\nHit: **{hit}**\nP&L: **${pnl:.2f}**",
            inline=False
        )
    embed.set_footer(text=f"Total closed trades in DB: {len(rows)} • $1k per trade simulation")
    await ctx.send(embed=embed)

bot.run(TOKEN)
