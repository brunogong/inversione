"""
Scanner autonomo — gira su GitHub Actions ogni 30 minuti.
Scansiona il mercato e invia segnali su Telegram.
NON richiede Streamlit.
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import httpx
import ta
from datetime import datetime, timezone, timedelta


# ═══════════════════════════════════════════════════════════════
# CONFIGURAZIONE (legge da environment variables di GitHub Actions)
# ═══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ACCOUNT_BALANCE = float(os.environ.get("ACCOUNT_BALANCE", "10000"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "1.5"))
MIN_CONFLUENCE = float(os.environ.get("MIN_CONFLUENCE", "4.5"))
MIN_RR_RATIO = 2.0

PAIRS = [
    "XAUUSD", "XAGUSD",
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "EURGBP", "EURAUD", "EURCHF", "EURNZD", "EURCAD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    "AUDNZD", "AUDCAD", "AUDJPY", "NZDJPY", "NZDCAD",
    "CADJPY", "CHFJPY",
]

SPECIAL_TICKERS = {"XAUUSD": "XAUUSD=X", "XAGUSD": "XAGUSD=X"}

WEIGHTS = {
    "ema_trend": 2.0, "ema_cross": 1.5, "rsi_reversal": 1.0,
    "rsi_divergence": 2.0, "macd_cross": 1.5, "macd_histogram": 1.0,
    "bollinger": 1.0, "stochastic": 1.0, "adx_strength": 0.5,
    "ichimoku": 1.5, "market_structure": 2.5, "candle_pattern": 1.0,
    "support_resistance": 1.5, "volume_confirm": 0.5,
}

SESSIONS = {
    "tokyo": {"start": 0, "end": 9},
    "london": {"start": 7, "end": 16},
    "newyork": {"start": 12, "end": 21},
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def get_pip_size(pair):
    if pair == "XAUUSD":
        return 0.01
    elif pair == "XAGUSD":
        return 0.001
    elif "JPY" in pair:
        return 0.01
    return 0.0001


def get_pip_value(pair, price):
    if pair in ["XAUUSD", "XAGUSD"]:
        return 1.0
    quote = pair[3:6]
    if quote == "USD":
        return 10.0
    elif "JPY" in pair:
        return (10.0 / price) * 100 if price > 0 else 10.0
    return 10.0 / price if price > 0 else 10.0


def get_ticker(pair):
    if pair in SPECIAL_TICKERS:
        return SPECIAL_TICKERS[pair]
    return f"{pair[:3]}{pair[3:]}=X"


def get_decimals(pair):
    if pair == "XAUUSD":
        return 2
    elif pair == "XAGUSD":
        return 3
    elif "JPY" in pair:
        return 3
    return 5


def get_session(hour):
    for name, t in SESSIONS.items():
        if t["start"] <= hour < t["end"]:
            return name
    return "off_hours"


# ═══════════════════════════════════════════════════════════════
# DATA FEED
# ═══════════════════════════════════════════════════════════════
def fetch_data(pair, interval="1h", period="60d"):
    ticker = get_ticker(pair)
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        return data
    except Exception as e:
        print(f"  Error fetching {pair}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# INDICATORI
# ═══════════════════════════════════════════════════════════════
def compute_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    if "Volume" not in df.columns:
        df["Volume"] = 0

    # EMA
    for p in [8, 13, 21, 50, 100, 200]:
        df[f"EMA_{p}"] = ta.trend.ema_indicator(close, window=p)

    # RSI
    df["RSI"] = ta.momentum.rsi(close, window=14)
    df["RSI_prev"] = df["RSI"].shift(1)

    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    df["MACD_hist_prev"] = df["MACD_hist"].shift(1)

    # Bollinger
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_pct"] = bb.bollinger_pband()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()

    # ADX
    adx_i = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx_i.adx()
    df["DI_plus"] = adx_i.adx_pos()
    df["DI_minus"] = adx_i.adx_neg()

    # ATR
    df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)

    # Ichimoku
    ichi = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    df["ICHI_tenkan"] = ichi.ichimoku_conversion_line()
    df["ICHI_kijun"] = ichi.ichimoku_base_line()
    df["ICHI_A"] = ichi.ichimoku_a()
    df["ICHI_B"] = ichi.ichimoku_b()

    # CCI + Williams
    df["CCI"] = ta.trend.cci(high, low, close, window=20)
    df["WILLR"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # Market Structure
    df = _market_structure(df)

    # Candle Patterns
    df = _candle_patterns(df)

    # S/R
    df["Resistance"] = df["High"].rolling(window=20).max()
    df["Support"] = df["Low"].rolling(window=20).min()
    atr = df["ATR"]
    threshold = atr * 0.5
    df["Near_Support"] = ((df["Close"] - df["Support"]).abs() < threshold).astype(int)
    df["Near_Resistance"] = ((df["Close"] - df["Resistance"]).abs() < threshold).astype(int)

    # RSI Divergence
    df = _rsi_divergence(df)

    return df


def _market_structure(df, lookback=5):
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            swing_high[i] = highs[i]
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            swing_low[i] = lows[i]

    df["Swing_High"] = swing_high
    df["Swing_Low"] = swing_low

    last_sh, last_sl = np.nan, np.nan
    structure = np.zeros(n)
    for i in range(n):
        if not np.isnan(swing_high[i]):
            if not np.isnan(last_sh):
                structure[i] = 1 if swing_high[i] > last_sh else -1
            last_sh = swing_high[i]
        if not np.isnan(swing_low[i]):
            if not np.isnan(last_sl):
                if swing_low[i] > last_sl:
                    structure[i] = max(structure[i], 1)
                else:
                    structure[i] = min(structure[i], -1)
            last_sl = swing_low[i]

    df["Structure"] = structure
    struct_s = pd.Series(structure)
    prev_s = struct_s.replace(0, np.nan).ffill()
    sb = np.zeros(n)
    for i in range(1, n):
        if structure[i] != 0 and not np.isnan(prev_s.iloc[i - 1]):
            if structure[i] != prev_s.iloc[i - 1]:
                sb[i] = structure[i]
    df["Structure_Break"] = sb
    return df


def _candle_patterns(df):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    pattern = np.zeros(n)
    body = np.abs(c - o)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l

    for i in range(2, n):
        avg_body = np.mean(body[max(0, i - 20):i]) if i > 0 else body[i]
        if avg_body == 0:
            continue
        if c[i-1] < o[i-1] and c[i] > o[i] and o[i] <= c[i-1] and c[i] >= o[i-1] and body[i] > avg_body * 1.2:
            pattern[i] = 1
        elif c[i-1] > o[i-1] and c[i] < o[i] and o[i] >= c[i-1] and c[i] <= o[i-1] and body[i] > avg_body * 1.2:
            pattern[i] = -1
        if lower_wick[i] > body[i] * 2 and upper_wick[i] < body[i] * 0.5 and body[i] > avg_body * 0.3:
            pattern[i] = max(pattern[i], 1)
        elif upper_wick[i] > body[i] * 2 and lower_wick[i] < body[i] * 0.5 and body[i] > avg_body * 0.3:
            pattern[i] = min(pattern[i], -1)
        if i >= 2:
            if c[i-2] < o[i-2] and body[i-1] < avg_body * 0.5 and c[i] > o[i] and c[i] > (o[i-2]+c[i-2])/2:
                pattern[i] = max(pattern[i], 1)
            if c[i-2] > o[i-2] and body[i-1] < avg_body * 0.5 and c[i] < o[i] and c[i] < (o[i-2]+c[i-2])/2:
                pattern[i] = min(pattern[i], -1)
    df["Candle_Pattern"] = pattern
    return df


def _rsi_divergence(df, lookback=14):
    close = df["Close"].values
    rsi = df["RSI"].values
    n = len(df)
    divergence = np.zeros(n)
    for i in range(lookback * 2, n):
        wc = close[i - lookback:i + 1]
        wr = rsi[i - lookback:i + 1]
        if np.any(np.isnan(wr)):
            continue
        if np.argmin(wc) == len(wc) - 1 and len(wc) > 3:
            if wc[-1] < np.min(wc[:-3]) and wr[-1] > np.min(wr[:-3]):
                divergence[i] = 1
        if np.argmax(wc) == len(wc) - 1 and len(wc) > 3:
            if wc[-1] > np.max(wc[:-3]) and wr[-1] < np.max(wr[:-3]):
                divergence[i] = -1
    df["RSI_Divergence"] = divergence
    return df


# ═══════════════════════════════════════════════════════════════
# STRATEGIA
# ═══════════════════════════════════════════════════════════════
def analyze_pair(df, pair, htf_bias=None):
    if len(df) < 200:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    scores = {}
    details = {}

    def _g(name, default=np.nan):
        v = last.get(name, default)
        if isinstance(v, float) and np.isnan(v):
            return default
        return v

    def _gp(name, default=np.nan):
        v = prev.get(name, default)
        if isinstance(v, float) and np.isnan(v):
            return default
        return v

    close = _g("Close")

    # 1. EMA TREND
    e50, e200 = _g("EMA_50"), _g("EMA_200")
    if e50 is not np.nan and e200 is not np.nan:
        scores["ema_trend"] = 1 if e50 > e200 else -1
        details["ema_trend"] = f"EMA 50 {'>' if e50 > e200 else '<'} 200"

    # 2. EMA CROSS
    e8, e21, pe8, pe21 = _g("EMA_8"), _g("EMA_21"), _gp("EMA_8"), _gp("EMA_21")
    if not any(v is np.nan for v in [e8, e21, pe8, pe21]):
        if pe8 <= pe21 and e8 > e21:
            scores["ema_cross"] = 1
            details["ema_cross"] = "EMA 8/21 Bullish cross ↑"
        elif pe8 >= pe21 and e8 < e21:
            scores["ema_cross"] = -1
            details["ema_cross"] = "EMA 8/21 Bearish cross ↓"
        else:
            scores["ema_cross"] = 0

    # 3. RSI
    rsi = _g("RSI", 50)
    rsi_p = _gp("RSI", 50)
    if rsi < 30:
        scores["rsi_reversal"] = 1
        details["rsi_reversal"] = f"RSI {rsi:.1f} Oversold"
    elif rsi > 70:
        scores["rsi_reversal"] = -1
        details["rsi_reversal"] = f"RSI {rsi:.1f} Overbought"
    elif rsi_p < 30 and rsi > 30:
        scores["rsi_reversal"] = 1
        details["rsi_reversal"] = "RSI leaving oversold ↑"
    elif rsi_p > 70 and rsi < 70:
        scores["rsi_reversal"] = -1
        details["rsi_reversal"] = "RSI leaving overbought ↓"
    else:
        scores["rsi_reversal"] = 0

    # 4. RSI DIVERGENCE
    rd = _g("RSI_Divergence", 0)
    scores["rsi_divergence"] = int(rd) if rd != 0 else 0
    if rd > 0:
        details["rsi_divergence"] = "Bullish RSI divergence 🟢"
    elif rd < 0:
        details["rsi_divergence"] = "Bearish RSI divergence 🔴"

    # 5. MACD CROSS
    m, ms, pm, pms = _g("MACD", 0), _g("MACD_signal", 0), _gp("MACD", 0), _gp("MACD_signal", 0)
    if not any(v is np.nan for v in [m, ms, pm, pms]):
        if pm <= pms and m > ms:
            scores["macd_cross"] = 1
            details["macd_cross"] = "MACD Bullish cross ↑"
        elif pm >= pms and m < ms:
            scores["macd_cross"] = -1
            details["macd_cross"] = "MACD Bearish cross ↓"
        else:
            scores["macd_cross"] = 0

    # 6. MACD HISTOGRAM
    mh, mhp = _g("MACD_hist", 0), _g("MACD_hist_prev", 0)
    if mhp < 0 < mh:
        scores["macd_histogram"] = 1
        details["macd_histogram"] = "MACD Hist turning bullish"
    elif mhp > 0 > mh:
        scores["macd_histogram"] = -1
        details["macd_histogram"] = "MACD Hist turning bearish"
    else:
        scores["macd_histogram"] = 0

    # 7. BOLLINGER
    bbp = _g("BB_pct", 0.5)
    if bbp < 0.05:
        scores["bollinger"] = 1
        details["bollinger"] = "Lower Bollinger — reversal"
    elif bbp > 0.95:
        scores["bollinger"] = -1
        details["bollinger"] = "Upper Bollinger — reversal"
    else:
        scores["bollinger"] = 0

    # 8. STOCHASTIC
    sk, sd, psk, psd = _g("STOCH_K", 50), _g("STOCH_D", 50), _gp("STOCH_K", 50), _gp("STOCH_D", 50)
    if sk < 20 and psk <= psd and sk > sd:
        scores["stochastic"] = 1
        details["stochastic"] = f"Stoch oversold cross ({sk:.0f})"
    elif sk > 80 and psk >= psd and sk < sd:
        scores["stochastic"] = -1
        details["stochastic"] = f"Stoch overbought cross ({sk:.0f})"
    else:
        scores["stochastic"] = 0

    # 9. ADX
    adx = _g("ADX", 0)
    dip, dim = _g("DI_plus", 0), _g("DI_minus", 0)
    if adx > 20:
        scores["adx_strength"] = 1 if dip > dim else -1
        details["adx_strength"] = f"ADX {adx:.0f} {'Bull' if dip > dim else 'Bear'}"
    else:
        scores["adx_strength"] = 0

    # 10. ICHIMOKU
    ia, ib = _g("ICHI_A"), _g("ICHI_B")
    tk, kj = _g("ICHI_tenkan"), _g("ICHI_kijun")
    if not any(v is np.nan for v in [ia, ib, tk, kj]):
        ct, cb = max(ia, ib), min(ia, ib)
        if close > ct and tk > kj:
            scores["ichimoku"] = 1
            details["ichimoku"] = "Above Ichimoku cloud"
        elif close < cb and tk < kj:
            scores["ichimoku"] = -1
            details["ichimoku"] = "Below Ichimoku cloud"
        else:
            scores["ichimoku"] = 0

    # 11. MARKET STRUCTURE
    sb = _g("Structure_Break", 0)
    st_val = _g("Structure", 0)
    if sb != 0:
        scores["market_structure"] = int(sb)
        details["market_structure"] = f"⚡ CHoCH {'Bullish' if sb > 0 else 'Bearish'}"
    elif st_val != 0:
        scores["market_structure"] = int(st_val) * 0.5
        details["market_structure"] = "Higher H/L" if st_val > 0 else "Lower H/L"
    else:
        scores["market_structure"] = 0

    # 12. CANDLE PATTERN
    cp = _g("Candle_Pattern", 0)
    if cp != 0:
        scores["candle_pattern"] = int(cp)
        details["candle_pattern"] = f"{'Bullish' if cp > 0 else 'Bearish'} reversal 🕯"
    else:
        scores["candle_pattern"] = 0

    # 13. S/R
    ns, nr = _g("Near_Support", 0), _g("Near_Resistance", 0)
    if ns:
        scores["support_resistance"] = 1
        details["support_resistance"] = "Near Support"
    elif nr:
        scores["support_resistance"] = -1
        details["support_resistance"] = "Near Resistance"
    else:
        scores["support_resistance"] = 0

    # 14. VOLUME
    vol = _g("Volume", 0)
    if vol > 0:
        avg_vol = df["Volume"].tail(20).mean()
        if avg_vol > 0 and vol > avg_vol * 1.5:
            scores["volume_confirm"] = 1 if close > last["Open"] else -1
            details["volume_confirm"] = "High volume confirmation"
        else:
            scores["volume_confirm"] = 0
    else:
        scores["volume_confirm"] = 0

    # SCORE
    bull = bear = max_p = 0.0
    for key, val in scores.items():
        w = WEIGHTS.get(key, 1.0)
        max_p += w
        if val > 0:
            bull += val * w
        elif val < 0:
            bear += abs(val) * w

    if bull > bear:
        direction, confluence = "BUY", bull
    elif bear > bull:
        direction, confluence = "SELL", bear
    else:
        return None

    if htf_bias is not None:
        if (direction == "BUY" and htf_bias < 0) or (direction == "SELL" and htf_bias > 0):
            confluence *= 0.6

    if confluence < MIN_CONFLUENCE:
        return None

    now = datetime.now(timezone.utc)
    session = get_session(now.hour)
    if session == "off_hours":
        confluence *= 0.8
        if confluence < MIN_CONFLUENCE:
            return None

    active = {}
    for key, val in scores.items():
        if val != 0 and key in details:
            active[key] = {"signal": "BULL" if val > 0 else "BEAR", "detail": details[key]}

    pct = confluence / max_p * 100 if max_p > 0 else 0
    if pct > 70:
        strength = "🔥 VERY STRONG"
    elif pct > 55:
        strength = "💪 STRONG"
    elif pct > 40:
        strength = "📊 MODERATE"
    else:
        strength = "⚠️ WEAK"

    atr = _g("ATR", 0)
    if atr == 0 or atr is np.nan:
        return None

    entry = close
    if direction == "BUY":
        sl = entry - atr * 1.5
        tp1, tp2, tp3 = entry + atr * 2, entry + atr * 3, entry + atr * 5
    else:
        sl = entry + atr * 1.5
        tp1, tp2, tp3 = entry - atr * 2, entry - atr * 3, entry - atr * 5

    sl_dist = abs(entry - sl)
    tp1_dist = abs(tp1 - entry)
    rr = tp1_dist / sl_dist if sl_dist > 0 else 0
    if rr < MIN_RR_RATIO:
        return None

    pip_sz = get_pip_size(pair)
    pip_val = get_pip_value(pair, entry)
    sl_pips = sl_dist / pip_sz

    # Lot size
    risk_amt = ACCOUNT_BALANCE * (RISK_PER_TRADE / 100)
    lot_size = risk_amt / (sl_pips * pip_val) if (sl_pips * pip_val) > 0 else 0
    lot_size = max(0.01, round(lot_size, 2))
    actual_risk = lot_size * sl_pips * pip_val
    dec = get_decimals(pair)

    return {
        "pair": pair, "direction": direction,
        "entry": round(entry, dec), "sl": round(sl, dec),
        "tp1": round(tp1, dec), "tp2": round(tp2, dec), "tp3": round(tp3, dec),
        "lot_size": lot_size, "risk": round(actual_risk, 2),
        "risk_pct": round((actual_risk / ACCOUNT_BALANCE) * 100, 2),
        "rr": round(rr, 2),
        "sl_pips": round(sl_pips, 1),
        "tp1_pips": round(tp1_dist / pip_sz, 1),
        "confluence": round(confluence, 2), "max_score": round(max_p, 2),
        "strength": strength, "session": session,
        "indicators": active,
        "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
    }


# ═══════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════
def send_telegram(signal):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ⚠️ Telegram non configurato")
        return False

    s = signal
    emoji = "🟢 BUY LONG" if s["direction"] == "BUY" else "🔴 SELL SHORT"
    fire = "🔥" if "STRONG" in s["strength"] else "📊"

    ind_lines = []
    for key, info in s["indicators"].items():
        icon = "✅" if (
            (s["direction"] == "BUY" and info["signal"] == "BULL") or
            (s["direction"] == "SELL" and info["signal"] == "BEAR")
        ) else "⬜"
        ind_lines.append(f"  {icon} {info['detail']}")
    ind_text = "\n".join(ind_lines) or "  N/A"

    msg = f"""{fire} <b>FOREX SIGNAL — {s['strength']}</b> {fire}
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 <b>{s['pair']}</b> | <b>{emoji}</b>
⏰ H1 | Session: {s['session'].upper()}
📈 Confluence: <b>{s['confluence']}/{s['max_score']}</b>

💰 <b>TRADE SETUP:</b>
  ▫️ Entry:  <code>{s['entry']}</code>
  ▫️ SL:     <code>{s['sl']}</code>  ({s['sl_pips']} pips)
  ▫️ TP1:    <code>{s['tp1']}</code>  ({s['tp1_pips']} pips)
  ▫️ TP2:    <code>{s['tp2']}</code>
  ▫️ TP3:    <code>{s['tp3']}</code>

📐 <b>RISK:</b>
  ▫️ Lots: <b>{s['lot_size']}</b>
  ▫️ Risk: ${s['risk']} ({s['risk_pct']}%)
  ▫️ R:R:  1:{s['rr']}

📊 <b>INDICATORS ({len(s['indicators'])}):</b>
{ind_text}

⏱ {s['timestamp']}
━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 <i>Forex Reversal Engine v2.0 (Auto)</i>"""

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        with httpx.Client(timeout=10) as client:
            r = client.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID, "text": msg,
                "parse_mode": "HTML", "disable_web_page_preview": True,
            })
            return r.status_code == 200
    except Exception as e:
        print(f"  Telegram error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# MAIN SCANNER
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(f"⚡ FOREX REVERSAL SCANNER — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Pairs: {len(PAIRS)} | Min Confluence: {MIN_CONFLUENCE}")
    print(f"   Balance: ${ACCOUNT_BALANCE} | Risk: {RISK_PER_TRADE}%")
    print("=" * 60)

    signals_found = []

    for pair in PAIRS:
        print(f"\n📊 Scanning {pair}...")

        # Fetch H1 data
        df = fetch_data(pair, "1h", "60d")
        if df is None or len(df) < 200:
            print(f"  ⚠️ Insufficient data for {pair}")
            continue

        # Compute indicators
        df = compute_indicators(df)

        # Get HTF bias from Daily
        htf_bias = None
        df_d = fetch_data(pair, "1d", "365d")
        if df_d is not None and len(df_d) > 50:
            df_d = compute_indicators(df_d)
            last_d = df_d.iloc[-1]
            e50_d = last_d.get("EMA_50", np.nan)
            e200_d = last_d.get("EMA_200", np.nan)
            if not np.isnan(e50_d) and not np.isnan(e200_d):
                htf_bias = 1 if e50_d > e200_d else -1
                print(f"  HTF Bias: {'Bullish ↑' if htf_bias > 0 else 'Bearish ↓'}")

        # Analyze
        signal = analyze_pair(df, pair, htf_bias)

        if signal:
            signals_found.append(signal)
            print(f"  🎯 SIGNAL: {pair} {signal['direction']} | Score: {signal['confluence']}/{signal['max_score']} | Lots: {signal['lot_size']}")

            # Send to Telegram
            sent = send_telegram(signal)
            print(f"  📤 Telegram: {'✅ Sent' if sent else '❌ Failed'}")
        else:
            print(f"  — No signal")

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ SCAN COMPLETE — {len(signals_found)} signals found")
    if signals_found:
        for s in signals_found:
            print(f"   🎯 {s['pair']} {s['direction']} | {s['strength']} | Score: {s['confluence']}")
    print("=" * 60)

    # Send summary if signals found
    if signals_found and TELEGRAM_TOKEN:
        summary = f"📊 <b>Scan Summary</b>\n⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n\n"
        summary += f"Pairs scanned: {len(PAIRS)}\n"
        summary += f"Signals found: {len(signals_found)}\n\n"
        for s in signals_found:
            emoji = "🟢" if s["direction"] == "BUY" else "🔴"
            summary += f"{emoji} {s['pair']} {s['direction']} — {s['strength']}\n"

        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            with httpx.Client(timeout=10) as client:
                client.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID, "text": summary,
                    "parse_mode": "HTML",
                })
        except:
            pass

    return len(signals_found)


if __name__ == "__main__":
    main()
