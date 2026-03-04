# strategies.py
import yfinance as yf
import pandas as pd
import numpy as np
from claude_ai import ask_claude
from config import LOOKBACK_PERIOD, MIN_VOLUME_INC, USE_CLAUDE_AI

def fetch_data(pair, timeframe, period="60d"):
    """Scarica dati dal mercato (YFinance)."""
    symbol = f"{pair}=X"
    df = yf.download(
        symbol,
        period=period,
        interval=timeframe,
        progress=False
    )
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def calculate_atr(df, period=14):
    """Calcola l'ATR (Average True Range)."""
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(period).mean()
    return df

def detect_market_structure_shift(df):
    """Rileva Market Structure Shift (MSS) - Higher High / Lower Low rotto."""
    df['swing_high'] = df['High'].rolling(5, center=True).max()
    df['swing_low']  = df['Low'].rolling(5, center=True).min()
    swings = df.dropna(subset=['swing_high', 'swing_low']).tail(5)
    
    if len(swings) < 3: 
        return None

    # **Uptrend → Inversione RIBASSISTA**
    if (swings['swing_high'].iloc[-2] < swings['swing_high'].iloc[-3]) and \
       (swings['swing_low'].iloc[-2] < swings['swing_low'].iloc[-3]):
        last_low = swings['swing_low'].iloc[-2]
        if df['Low'].iloc[-1] < last_low:
            return "SELL"
    
    # **Downtrend → Inversione RIALZISTA**
    if (swings['swing_high'].iloc[-2] > swings['swing_high'].iloc[-3]) and \
       (swings['swing_low'].iloc[-2] > swings['swing_low'].iloc[-3]):
        last_high = swings['swing_high'].iloc[-2]
        if df['High'].iloc[-1] > last_high:
            return "BUY"
    
    return None

def detect_choch(df):
    """Rileva Change of Character (CHOCH) secondo ICT/Smart Money."""
    df = calculate_atr(df)
    last_candles = df.tail(5)
    atr = df['ATR'].iloc[-1]
    
    # CHOCH BULLISH
    if (last_candles['Close'].iloc[-3] < last_candles['Close'].iloc[-4] and
        last_candles['Close'].iloc[-2] < last_candles['Close'].iloc[-3] and
        last_candles['Close'].iloc[-1] > last_candles['High'].iloc[-2] and
        (last_candles['High'].iloc[-1] - last_candles['Low'].iloc[-1]) > 1.5 * atr):
        
        # Controlla volume
        vol_avg = df['Volume'].tail(20).mean()
        if last_candles['Volume'].iloc[-1] > vol_avg * MIN_VOLUME_INC:
            return "BUY"
    
    # CHOCH BEARISH
    if (last_candles['Close'].iloc[-3] > last_candles['Close'].iloc[-4] and
        last_candles['Close'].iloc[-2] > last_candles['Close'].iloc[-3] and
        last_candles['Close'].iloc[-1] < last_candles['Low'].iloc[-2] and
        (last_candles['High'].iloc[-1] - last_candles['Low'].iloc[-1]) > 1.5 * atr):
        
        vol_avg = df['Volume'].tail(20).mean()
        if last_candles['Volume'].iloc[-1] > vol_avg * MIN_VOLUME_INC:
            return "SELL"
    
    return None

def detect_fvg(df):
    """Rileva Fair Value Gap (FVG) - Concetto ICT."""
    for i in range(2, LOOKBACK_PERIOD):
        # FVG Bullish
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            return "BUY"
        # FVG Bearish
        if df['High'].iloc[i] < df['Low'].iloc[i-2]:
            return "SELL"
    return None

def detect_divergence(df):
    """Rileva divergenze RSI (14 periodi)."""
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                            df['Close'].diff().clip(upper=0).rolling(14).mean().abs()))
    
    # Divergenza RIALZISTA (Prezzo fa lower low, RSI fa higher low)
    if df['Low'].iloc[-3] < df['Low'].iloc[-5] and df['RSI'].iloc[-3] > df['RSI'].iloc[-5]:
        return "BUY"
    
    # Divergenza RIBASSISTA (Prezzo fa higher high, RSI fa lower high)
    if df['High'].iloc[-3] > df['High'].iloc[-5] and df['RSI'].iloc[-3] < df['RSI'].iloc[-5]:
        return "SELL"
    
    return None

def analyze_pair(pair, timeframe):
    """Analizza una coppia su un timeframe. Restituisce il segnale se confermato."""
    df = fetch_data(pair, timeframe)
    if df.empty: 
        return None
    
    df = calculate_atr(df)
    
    mss  = detect_market_structure_shift(df)
    choch = detect_choch(df)
    fvg  = detect_fvg(df)
    div  = detect_divergence(df)
    
    signals = []
    if mss:  signals.append(mss)
    if choch: signals.append(choch)
    if fvg:  signals.append(fvg)
    if div:  signals.append(div)
    
    # Almeno 2 strategie devono concordare
    if len(signals) < 2: 
        return None
    
    final_signal = max(set(signals), key=signals.count)  # Segnale più frequente
    
    # **CONFERMA con CLAUDE AI** (se abilitato)
    if USE_CLAUDE_AI:
        context = f"""
        Coppia: {pair} | Timeframe: {timeframe}
        Ultimo prezzo: {df['Close'].iloc[-1]:.5f}
        Segnale rilevato: {final_signal}
        Market Structure Shift: {mss}
        CHOCH: {choch}
        FVG: {fvg}
        Divergenza RSI: {div}
        ATR: {df['ATR'].iloc[-1]:.5f}
        Volume ultimo: {df['Volume'].iloc[-1]} (Media ultimi 20: {df['Volume'].tail(20).mean():.0f})
        """
        ai_verdict = ask_claude(context)
        if ai_verdict == "NO_SIGNAL":
            return None
        final_signal = ai_verdict  # Accetta il giudizio di Claude
    
    return final_signal
