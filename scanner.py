"""
Scanner autonomo — gira su GitHub Actions.
Versione robusta con gestione errori completa.
"""
import os
import sys
import traceback

# Installa dipendenze se mancanti
try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import httpx
    import ta
except ImportError as e:
    print(f"Installing missing dependency: {e}")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "numpy", "pandas", "yfinance", "ta", "httpx"])
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import httpx
    import ta

from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ACCOUNT_BALANCE = 2000
RISK_PER_TRADE = 1.5
MIN_CONFLUENCE = 4.5
MIN_RR_RATIO = 2.0

PAIRS = [
    "XAUUSD", "XAGUSD",
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "EURGBP", "EURAUD", "EURCHF",
    "GBPJPY", "GBPCHF",
    "AUDNZD", "AUDJPY", "CADJPY",
]

SPECIAL_TICKERS = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
}

WEIGHTS = {
    "ema_trend": 2.0, "ema_cross": 1.5, "rsi_reversal": 1.0,
    "rsi_divergence": 2.0, "macd_cross": 1.5, "macd_histogram": 1.0,
    "bollinger": 1.0, "stochastic": 1.0, "adx_strength": 0.5,
    "ichimoku": 1.5, "market_structure": 2.5, "candle_pattern": 1.0,
    "support_resistance": 1.5, "volume_confirm": 0.5,
}

SESSIONS = {
    "tokyo": (0, 9), "london": (7, 16), "newyork": (12, 21),
}


def get_pip_size(pair):
    if pair == "XAUUSD": return 0.10
    if pair == "XAGUSD": return 0.01
    if "JPY" in pair: return 0.01
    return 0.0001

def get_pip_value(pair, price):
    if pair in ["XAUUSD", "XAGUSD"]: return 1.0
    if pair[3:6] == "USD": return 10.0
    if "JPY" in pair: return (10.0 / price) * 100 if price > 0 else 10.0
    return 10.0 / price if price > 0 else 10.0

def get_decimals(pair):
    if pair == "XAUUSD": return 2
    if pair == "XAGUSD": return 3
    if "JPY" in pair: return 3
    return 5

def get_ticker(pair):
    if pair in SPECIAL_TICKERS:
        return SPECIAL_TICKERS[pair]
    return f"{pair[:3]}{pair[3:]}=X"

def get_session(hour):
    for name, (s, e) in SESSIONS.items():
        if s <= hour < e:
            return name
    return "off_hours"


# ═══════════════════════════════════════════════════════════════
# DATA FEED
# ═══════════════════════════════════════════════════════════════
def fetch_data(pair, interval="1h", period="60d"):
    ticker = get_ticker(pair)
    try:
        print(f"    Downloading {ticker}...", end=" ")
        data = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=True,
        )

        if data is None or data.empty:
            print("NO DATA")
            return None

        # Fix multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Standardize column names
        col_map = {}
        for col in data.columns:
            col_lower = str(col).lower()
            if "open" in col_lower: col_map[col] = "Open"
            elif "high" in col_lower: col_map[col] = "High"
            elif "low" in col_lower: col_map[col] = "Low"
            elif "close" in col_lower: col_map[col] = "Close"
            elif "volume" in col_lower: col_map[col] = "Volume"
        data = data.rename(columns=col_map)

        # Verifica colonne necessarie
        required = ["Open", "High", "Low", "Close"]
        for col in required:
            if col not in data.columns:
                print(f"MISSING COLUMN: {col}")
                return None

        if "Volume" not in data.columns:
            data["Volume"] = 0

        # Rimuovi righe con NaN nelle colonne principali
        data = data.dropna(subset=required)

        print(f"OK ({len(data)} candles)")
        return data

    except Exception as e:
        print(f"ERROR: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# INDICATORI
# ═══════════════════════════════════════════════════════════════
def compute_indicators(df):
    try:
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        # EMA
        for p in [8, 21, 50, 200]:
            try:
                df[f"EMA_{p}"] = ta.trend.ema_indicator(close, window=p)
            except:
                df[f"EMA_{p}"] = np.nan

        # RSI
        try:
            df["RSI"] = ta.momentum.rsi(close, window=14)
            df["RSI_prev"] = df["RSI"].shift(1)
        except:
            df["RSI"] = 50.0
            df["RSI_prev"] = 50.0

        # MACD
        try:
            macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
            df["MACD"] = macd_ind.macd()
            df["MACD_signal"] = macd_ind.macd_signal()
            df["MACD_hist"] = macd_ind.macd_diff()
            df["MACD_hist_prev"] = df["MACD_hist"].shift(1)
        except:
            df["MACD"] = 0
            df["MACD_signal"] = 0
            df["MACD_hist"] = 0
            df["MACD_hist_prev"] = 0

        # Bollinger
        try:
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            df["BB_pct"] = bb.bollinger_pband()
        except:
            df["BB_pct"] = 0.5

        # Stochastic
        try:
            stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            df["STOCH_K"] = stoch.stoch()
            df["STOCH_D"] = stoch.stoch_signal()
        except:
            df["STOCH_K"] = 50.0
            df["STOCH_D"] = 50.0

        # ADX
        try:
            adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
            df["ADX"] = adx_ind.adx()
            df["DI_plus"] = adx_ind.adx_pos()
            df["DI_minus"] = adx_ind.adx_neg()
        except:
            df["ADX"] = 0
            df["DI_plus"] = 0
            df["DI_minus"] = 0

        # ATR
        try:
            df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)
        except:
            df["ATR"] = (high - low).rolling(14).mean()

        # Ichimoku
        try:
            ichi = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
            df["ICHI_tenkan"] = ichi.ichimoku_conversion_line()
            df["ICHI_kijun"] = ichi.ichimoku_base_line()
            df["ICHI_A"] = ichi.ichimoku_a()
            df["ICHI_B"] = ichi.ichimoku_b()
        except:
            df["ICHI_tenkan"] = np.nan
            df["ICHI_kijun"] = np.nan
            df["ICHI_A"] = np.nan
            df["ICHI_B"] = np.nan

        # Market Structure
        try:
            df = market_structure(df)
        except:
            df["Structure"] = 0
            df["Structure_Break"] = 0

        # Candle Patterns
        try:
            df = candle_patterns(df)
        except:
            df["Candle_Pattern"] = 0

        # S/R
        try:
            df["Resistance"] = high.rolling(window=20).max()
            df["Support"] = low.rolling(window=20).min()
            atr_val = df["ATR"]
            threshold = atr_val * 0.5
            df["Near_Support"] = ((close - df["Support"]).abs() < threshold).astype(int)
            df["Near_Resistance"] = ((close - df["Resistance"]).abs() < threshold).astype(int)
        except:
            df["Near_Support"] = 0
            df["Near_Resistance"] = 0

        # RSI Divergence
        try:
            df = rsi_divergence(df)
        except:
            df["RSI_Divergence"] = 0

        return df

    except Exception as e:
        print(f"    Indicator error: {e}")
        return df


def market_structure(df, lookback=5):
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(df)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == np.max(window_h):
            swing_high[i] = highs[i]
        if lows[i] == np.min(window_l):
            swing_low[i] = lows[i]

    df["Swing_High"] = swing_high
    df["Swing_Low"] = swing_low

    last_sh = np.nan
    last_sl = np.nan
    structure = np.zeros(n)

    for i in range(n):
        if not np.isnan(swing_high[i]):
            if not np.isnan(last_sh):
                structure[i] = 1.0 if swing_high[i] > last_sh else -1.0
            last_sh = swing_high[i]
        if not np.isnan(swing_low[i]):
            if not np.isnan(last_sl):
                if swing_low[i] > last_sl:
                    structure[i] = max(structure[i], 1.0)
                else:
                    structure[i] = min(structure[i], -1.0)
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


def candle_patterns(df):
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    n = len(df)
    pattern = np.zeros(n)
    body = np.abs(c - o)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l

    for i in range(2, n):
        avg_body = np.mean(body[max(0, i - 20):i])
        if avg_body == 0:
            continue

        # Engulfing
        if (c[i-1] < o[i-1] and c[i] > o[i] and
            o[i] <= c[i-1] and c[i] >= o[i-1] and body[i] > avg_body * 1.2):
            pattern[i] = 1
        elif (c[i-1] > o[i-1] and c[i] < o[i] and
              o[i] >= c[i-1] and c[i] <= o[i-1] and body[i] > avg_body * 1.2):
            pattern[i] = -1

        # Hammer / Shooting star
        if body[i] > 0:
            if (lower_wick[i] > body[i] * 2 and upper_wick[i] < body[i] * 0.5
                and body[i] > avg_body * 0.3):
                pattern[i] = max(pattern[i], 1)
            elif (upper_wick[i] > body[i] * 2 and lower_wick[i] < body[i] * 0.5
                  and body[i] > avg_body * 0.3):
                pattern[i] = min(pattern[i], -1)

        # Morning / Evening star
        if i >= 2:
            if (c[i-2] < o[i-2] and body[i-1] < avg_body * 0.5
                and c[i] > o[i] and c[i] > (o[i-2]+c[i-2])/2):
                pattern[i] = max(pattern[i], 1)
            if (c[i-2] > o[i-2] and body[i-1] < avg_body * 0.5
                and c[i] < o[i] and c[i] < (o[i-2]+c[i-2])/2):
                pattern[i] = min(pattern[i], -1)

    df["Candle_Pattern"] = pattern
    return df


def rsi_divergence(df, lookback=14):
    close = df["Close"].values.astype(float)
    rsi = df["RSI"].values.astype(float)
    n = len(df)
    divergence = np.zeros(n)

    for i in range(lookback * 2, n):
        wc = close[i - lookback:i + 1]
        wr = rsi[i - lookback:i + 1]
        if np.any(np.isnan(wr)) or np.any(np.isnan(wc)):
            continue
        if len(wc) > 3:
            if np.argmin(wc) == len(wc) - 1:
                if wc[-1] < np.min(wc[:-3]) and wr[-1] > np.min(wr[:-3]):
                    divergence[i] = 1
            if np.argmax(wc) == len(wc) - 1:
                if wc[-1] > np.max(wc[:-3]) and wr[-1] < np.max(wr[:-3]):
                    divergence[i] = -1

    df["RSI_Divergence"] = divergence
    return df


# ═══════════════════════════════════════════════════════════════
# STRATEGIA
# ═══════════════════════════════════════════════════════════════
def safe_get(row, name, default=np.nan):
    try:
        v = row.get(name, default)
        if v is None:
            return default
        if isinstance(v, (int, float)) and np.isnan(v):
            return default
        return float(v)
    except:
        return default


def analyze_pair(df, pair, htf_bias=None):
    try:
        if len(df) < 200:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        scores = {}
        details = {}

        close = safe_get(last, "Close", 0)
        if close == 0:
            return None

        # 1. EMA TREND
        e50 = safe_get(last, "EMA_50")
        e200 = safe_get(last, "EMA_200")
        if not np.isnan(e50) and not np.isnan(e200):
            scores["ema_trend"] = 1 if e50 > e200 else -1
            details["ema_trend"] = f"EMA 50{'>' if e50 > e200 else '<'}200"

        # 2. EMA CROSS
        e8 = safe_get(last, "EMA_8")
        e21 = safe_get(last, "EMA_21")
        pe8 = safe_get(prev, "EMA_8")
        pe21 = safe_get(prev, "EMA_21")
        if not any(np.isnan(v) for v in [e8, e21, pe8, pe21]):
            if pe8 <= pe21 and e8 > e21:
                scores["ema_cross"] = 1
                details["ema_cross"] = "EMA 8/21 Bull cross ↑"
            elif pe8 >= pe21 and e8 < e21:
                scores["ema_cross"] = -1
                details["ema_cross"] = "EMA 8/21 Bear cross ↓"
            else:
                scores["ema_cross"] = 0

        # 3. RSI
        rsi = safe_get(last, "RSI", 50)
        rsi_p = safe_get(prev, "RSI", 50)
        if rsi < 30:
            scores["rsi_reversal"] = 1
            details["rsi_reversal"] = f"RSI {rsi:.0f} Oversold"
        elif rsi > 70:
            scores["rsi_reversal"] = -1
            details["rsi_reversal"] = f"RSI {rsi:.0f} Overbought"
        elif rsi_p < 30 and rsi > 30:
            scores["rsi_reversal"] = 1
            details["rsi_reversal"] = "RSI leaving oversold ↑"
        elif rsi_p > 70 and rsi < 70:
            scores["rsi_reversal"] = -1
            details["rsi_reversal"] = "RSI leaving overbought ↓"
        else:
            scores["rsi_reversal"] = 0

        # 4. RSI DIVERGENCE
        rd = safe_get(last, "RSI_Divergence", 0)
        if rd > 0:
            scores["rsi_divergence"] = 1
            details["rsi_divergence"] = "Bull RSI divergence 🟢"
        elif rd < 0:
            scores["rsi_divergence"] = -1
            details["rsi_divergence"] = "Bear RSI divergence 🔴"
        else:
            scores["rsi_divergence"] = 0

        # 5. MACD CROSS
        m = safe_get(last, "MACD", 0)
        ms = safe_get(last, "MACD_signal", 0)
        pm = safe_get(prev, "MACD", 0)
        pms = safe_get(prev, "MACD_signal", 0)
        if pm <= pms and m > ms:
            scores["macd_cross"] = 1
            details["macd_cross"] = "MACD Bull cross ↑"
        elif pm >= pms and m < ms:
            scores["macd_cross"] = -1
            details["macd_cross"] = "MACD Bear cross ↓"
        else:
            scores["macd_cross"] = 0

        # 6. MACD HISTOGRAM
        mh = safe_get(last, "MACD_hist", 0)
        mhp = safe_get(last, "MACD_hist_prev", 0)
        if mhp < 0 < mh:
            scores["macd_histogram"] = 1
            details["macd_histogram"] = "MACD Hist bullish"
        elif mhp > 0 > mh:
            scores["macd_histogram"] = -1
            details["macd_histogram"] = "MACD Hist bearish"
        else:
            scores["macd_histogram"] = 0

        # 7. BOLLINGER
        bbp = safe_get(last, "BB_pct", 0.5)
        if bbp < 0.05:
            scores["bollinger"] = 1
            details["bollinger"] = "Lower BB reversal"
        elif bbp > 0.95:
            scores["bollinger"] = -1
            details["bollinger"] = "Upper BB reversal"
        else:
            scores["bollinger"] = 0

        # 8. STOCHASTIC
        sk = safe_get(last, "STOCH_K", 50)
        sd = safe_get(last, "STOCH_D", 50)
        psk = safe_get(prev, "STOCH_K", 50)
        psd = safe_get(prev, "STOCH_D", 50)
        if sk < 20 and psk <= psd and sk > sd:
            scores["stochastic"] = 1
            details["stochastic"] = f"Stoch oversold ({sk:.0f})"
        elif sk > 80 and psk >= psd and sk < sd:
            scores["stochastic"] = -1
            details["stochastic"] = f"Stoch overbought ({sk:.0f})"
        else:
            scores["stochastic"] = 0

        # 9. ADX
        adx = safe_get(last, "ADX", 0)
        dip = safe_get(last, "DI_plus", 0)
        dim = safe_get(last, "DI_minus", 0)
        if adx > 20:
            scores["adx_strength"] = 1 if dip > dim else -1
            details["adx_strength"] = f"ADX {adx:.0f}"
        else:
            scores["adx_strength"] = 0

        # 10. ICHIMOKU
        ia = safe_get(last, "ICHI_A")
        ib = safe_get(last, "ICHI_B")
        tk = safe_get(last, "ICHI_tenkan")
        kj = safe_get(last, "ICHI_kijun")
        if not any(np.isnan(v) for v in [ia, ib, tk, kj]):
            ct = max(ia, ib)
            cb = min(ia, ib)
            if close > ct and tk > kj:
                scores["ichimoku"] = 1
                details["ichimoku"] = "Above Ichimoku"
            elif close < cb and tk < kj:
                scores["ichimoku"] = -1
                details["ichimoku"] = "Below Ichimoku"
            else:
                scores["ichimoku"] = 0

        # 11. MARKET STRUCTURE
        sb = safe_get(last, "Structure_Break", 0)
        st_val = safe_get(last, "Structure", 0)
        if sb > 0:
            scores["market_structure"] = 1
            details["market_structure"] = "⚡ CHoCH Bullish"
        elif sb < 0:
            scores["market_structure"] = -1
            details["market_structure"] = "⚡ CHoCH Bearish"
        elif st_val > 0:
            scores["market_structure"] = 0.5
            details["market_structure"] = "Higher H/L"
        elif st_val < 0:
            scores["market_structure"] = -0.5
            details["market_structure"] = "Lower H/L"
        else:
            scores["market_structure"] = 0

        # 12. CANDLE
        cp = safe_get(last, "Candle_Pattern", 0)
        if cp > 0:
            scores["candle_pattern"] = 1
            details["candle_pattern"] = "Bull reversal 🕯"
        elif cp < 0:
            scores["candle_pattern"] = -1
            details["candle_pattern"] = "Bear reversal 🕯"
        else:
            scores["candle_pattern"] = 0

        # 13. S/R
        ns = safe_get(last, "Near_Support", 0)
        nr = safe_get(last, "Near_Resistance", 0)
        if ns > 0:
            scores["support_resistance"] = 1
            details["support_resistance"] = "Near Support"
        elif nr > 0:
            scores["support_resistance"] = -1
            details["support_resistance"] = "Near Resistance"
        else:
            scores["support_resistance"] = 0

        # 14. VOLUME
        vol = safe_get(last, "Volume", 0)
        if vol > 0:
            avg_vol = float(df["Volume"].tail(20).mean())
            if avg_vol > 0 and vol > avg_vol * 1.5:
                open_price = safe_get(last, "Open", close)
                scores["volume_confirm"] = 1 if close > open_price else -1
                details["volume_confirm"] = "High volume"
            else:
                scores["volume_confirm"] = 0
        else:
            scores["volume_confirm"] = 0

        # ═══ SCORE ═══
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

        # HTF bias
        if htf_bias is not None:
            if (direction == "BUY" and htf_bias < 0) or (direction == "SELL" and htf_bias > 0):
                confluence *= 0.6

        if confluence < MIN_CONFLUENCE:
            return None

        # Session
        now = datetime.now(timezone.utc)
        session = get_session(now.hour)
        if session == "off_hours":
            confluence *= 0.8
            if confluence < MIN_CONFLUENCE:
                return None

        # Active indicators
        active = {}
        for key, val in scores.items():
            if val != 0 and key in details:
                active[key] = {
                    "signal": "BULL" if val > 0 else "BEAR",
                    "detail": details[key],
                }

        # Strength
        pct = confluence / max_p * 100 if max_p > 0 else 0
        if pct > 70: strength = "🔥 VERY STRONG"
        elif pct > 55: strength = "💪 STRONG"
        elif pct > 40: strength = "📊 MODERATE"
        else: strength = "⚠️ WEAK"

        # ATR
        atr = safe_get(last, "ATR", 0)
        if atr <= 0:
            return None

        # Trade levels
        entry = close
        if direction == "BUY":
            sl = entry - atr * 1.5
            tp1 = entry + atr * 2
            tp2 = entry + atr * 3
            tp3 = entry + atr * 5
        else:
            sl = entry + atr * 1.5
            tp1 = entry - atr * 2
            tp2 = entry - atr * 3
            tp3 = entry - atr * 5

        sl_dist = abs(entry - sl)
        tp1_dist = abs(tp1 - entry)
        rr = tp1_dist / sl_dist if sl_dist > 0 else 0
        if rr < MIN_RR_RATIO:
            return None

        # Position sizing
        pip_sz = get_pip_size(pair)
        pip_val = get_pip_value(pair, entry)
        sl_pips = sl_dist / pip_sz

        risk_amt = ACCOUNT_BALANCE * (RISK_PER_TRADE / 100)
        denom = sl_pips * pip_val
        lot_size = risk_amt / denom if denom > 0 else 0
        lot_size = max(0.01, round(lot_size, 2))
        actual_risk = lot_size * sl_pips * pip_val
        dec = get_decimals(pair)

        return {
            "pair": pair, "direction": direction,
            "entry": round(entry, dec), "sl": round(sl, dec),
            "tp1": round(tp1, dec), "tp2": round(tp2, dec), "tp3": round(tp3, dec),
            "lot_size": lot_size, "risk": round(actual_risk, 2),
            "risk_pct": round((actual_risk / ACCOUNT_BALANCE) * 100, 2),
            "rr": round(rr, 2), "sl_pips": round(sl_pips, 1),
            "tp1_pips": round(tp1_dist / pip_sz, 1),
            "confluence": round(confluence, 2), "max_score": round(max_p, 2),
            "strength": strength, "session": session,
            "indicators": active,
            "timestamp": now.strftime("%Y-%m-%d %H:%M UTC"),
        }

    except Exception as e:
        print(f"    Analysis error for {pair}: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════
def send_telegram(signal):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("    ⚠️ Telegram not configured")
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

    msg = f"""{fire} <b>SIGNAL — {s['strength']}</b> {fire}
━━━━━━━━━━━━━━━━━━━━━
📊 <b>{s['pair']}</b> | <b>{emoji}</b>
⏰ H1 | {s['session'].upper()}
📈 Score: <b>{s['confluence']}/{s['max_score']}</b>

💰 <b>TRADE:</b>
  ▫️ Entry: <code>{s['entry']}</code>
  ▫️ SL:    <code>{s['sl']}</code> ({s['sl_pips']}p)
  ▫️ TP1:   <code>{s['tp1']}</code> ({s['tp1_pips']}p)
  ▫️ TP2:   <code>{s['tp2']}</code>
  ▫️ TP3:   <code>{s['tp3']}</code>

📐 <b>RISK:</b>
  ▫️ Lots: <b>{s['lot_size']}</b>
  ▫️ Risk: ${s['risk']} ({s['risk_pct']}%)
  ▫️ R:R:  1:{s['rr']}

📊 <b>INDICATORS ({len(s['indicators'])}):</b>
{ind_text}

⏱ {s['timestamp']}
━━━━━━━━━━━━━━━━━━━━━
🤖 <i>Auto Scanner v2.1</i>"""

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        with httpx.Client(timeout=15) as client:
            r = client.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID, "text": msg,
                "parse_mode": "HTML",
            })
            if r.status_code == 200:
                return True
            else:
                print(f"    Telegram error: {r.status_code} {r.text[:200]}")
                return False
    except Exception as e:
        print(f"    Telegram error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    now = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"⚡ FOREX SCANNER — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Pairs: {len(PAIRS)} | Min Score: {MIN_CONFLUENCE}")
    print(f"   Balance: ${ACCOUNT_BALANCE} | Risk: {RISK_PER_TRADE}%")
    print(f"   Telegram: {'✅' if TELEGRAM_TOKEN else '❌'}")
    print("=" * 60)

    signals_found = []

    for pair in PAIRS:
        print(f"\n📊 {pair}")

        try:
            # H1 data
            df = fetch_data(pair, "1h", "60d")
            if df is None or len(df) < 200:
                print(f"    ⚠️ Not enough data ({len(df) if df is not None else 0} candles)")
                continue

            # Indicators
            df = compute_indicators(df)

            # Daily bias
            htf_bias = None
            try:
                df_d = fetch_data(pair, "1d", "365d")
                if df_d is not None and len(df_d) > 50:
                    df_d = compute_indicators(df_d)
                    e50_d = safe_get(df_d.iloc[-1], "EMA_50")
                    e200_d = safe_get(df_d.iloc[-1], "EMA_200")
                    if not np.isnan(e50_d) and not np.isnan(e200_d):
                        htf_bias = 1 if e50_d > e200_d else -1
                        print(f"    HTF: {'Bull ↑' if htf_bias > 0 else 'Bear ↓'}")
            except Exception as e:
                print(f"    HTF error: {e}")

            # Analyze
            signal = analyze_pair(df, pair, htf_bias)

            if signal:
                signals_found.append(signal)
                print(f"    🎯 {signal['direction']} | Score: {signal['confluence']}/{signal['max_score']} | Lots: {signal['lot_size']}")
                sent = send_telegram(signal)
                print(f"    📤 Telegram: {'✅' if sent else '❌'}")
            else:
                print(f"    — No signal")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ DONE — {len(signals_found)} signals")
    for s in signals_found:
        print(f"   🎯 {s['pair']} {s['direction']} | {s['strength']}")
    print("=" * 60)

    # Telegram summary
    if signals_found and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            summary = f"📊 <b>Scan Complete</b> — {now.strftime('%H:%M UTC')}\n\n"
            summary += f"Scanned: {len(PAIRS)} pairs\n"
            summary += f"Found: {len(signals_found)} signals\n\n"
            for s in signals_found:
                e = "🟢" if s["direction"] == "BUY" else "🔴"
                summary += f"{e} {s['pair']} {s['direction']} — {s['strength']}\n"

            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            with httpx.Client(timeout=15) as client:
                client.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID, "text": summary,
                    "parse_mode": "HTML",
                })
        except:
            pass
    elif not signals_found and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        # Invia messaggio anche se non ci sono segnali (opzionale)
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            with httpx.Client(timeout=15) as client:
                client.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": f"📊 Scan {now.strftime('%H:%M UTC')} — {len(PAIRS)} pairs scanned, no signals",
                    "parse_mode": "HTML",
                })
        except:
            pass

    print("\n🏁 Scanner finished successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
