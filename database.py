# database.py
import sqlite3
from datetime import datetime

DB_NAME = "signals.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS signals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  pair TEXT,
                  direction TEXT,
                  entry REAL,
                  tp REAL,
                  sl REAL,
                  lot REAL,
                  timeframe TEXT,
                  atr REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_signal(signal_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''INSERT INTO signals 
                 (pair, direction, entry, tp, sl, lot, timeframe, atr, timestamp)
                 VALUES (?,?,?,?,?,?,?,?,?)''',
              (signal_data['pair'],
               signal_data['direction'],
               signal_data['entry'],
               signal_data['tp'],
               signal_data['sl'],
               signal_data['lot'],
               signal_data['timeframe'],
               signal_data['atr'],
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
              )
    conn.commit()
    conn.close()
