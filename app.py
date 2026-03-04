# app.py
from flask import Flask, render_template
import time
from datetime import datetime
import yfinance as yf
import numpy as np

from strategies import analyze_pair
from risk_management import calculate_lot_size, calculate_tp_sl
from telegram_bot import send_telegram_message, format_signal_message
from database import init_db, save_signal
from config import CURRENCY_PAIRS, TIMEFRAMES, ACCOUNT_EQUITY

app = Flask(__name__)
init_db()  # Inizializza il DB

def generate_signal(pair, timeframe):
    """Genera un segnale per una coppia e un timeframe."""
    direction = analyze_pair(pair, timeframe)
    if not direction:
        return None
    
    # Ottieni ultimo prezzo
    df = yf.download(f"{pair}=X", interval=timeframe, period="1d", progress=False)
    entry_price = round(df['Close'].iloc[-1], 5)
    
    # Calcola ATR
    df['TR'] = np.maximum(df['High'] - df['Low'],
                         np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                   abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
    atr = df['ATR'].iloc[-1]
    
    # Calcola TP, SL, SL_pips
    tp, sl, sl_pips = calculate_tp_sl(entry_price, direction, atr, pair)
    
    # Calcola lotto
    lot_size = calculate_lot_size(sl_pips, pair)
    
    signal_data = {
        "pair": pair,
        "direction": direction,
        "entry": entry_price,
        "tp": tp,
        "sl": sl,
        "sl_pips": sl_pips,
        "lot": lot_size,
        "timeframe": timeframe,
        "atr": atr,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Salva nel DB
    save_signal(signal_data)
    
    # Invia su Telegram
    msg = format_signal_message(signal_data)
    send_telegram_message(msg)
    
    return signal_data

@app.route('/')
def home():
    """Dashboard web che mostra gli ultimi segnali."""
    from database import sqlite3
    conn = sqlite3.connect("signals.db")
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 20", conn)
    conn.close()
    return render_template('index.html', signals=df.to_dict('records'))

def run_scanner():
    """Scansiona tutte le coppie e timeframe ogni 15 minuti."""
    while True:
        for pair in CURRENCY_PAIRS:
            for tf in TIMEFRAMES:
                generate_signal(pair, tf)
        time.sleep(900)  # 15 minuti = 900 secondi

if __name__ == "__main__":
    import threading
    # Avvia lo scanner in un thread separato
    scanner_thread = threading.Thread(target=run_scanner, daemon=True)
    scanner_thread.start()
    
    # Avvia la web app
    app.run(host='0.0.0.0', port=5000, debug=False)
