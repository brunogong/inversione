# risk_management.py
from config import RISK_PERCENT, ACCOUNT_EQUITY, RR_RATIO

def calculate_lot_size(sl_pips, pair, equity=None):
    """
    Calcola la dimensione del lotto.
    equity: capitale attuale (se None usa ACCOUNT_EQUITY)
    """
    if equity is None:
        equity = ACCOUNT_EQUITY
    
    pip_value = 10.0
    if "JPY" in pair: 
        pip_value = 0.83  # 1 pip = 0.83 USD per JPY pairs
    
    risk_amount = (equity * RISK_PERCENT) / 100
    lot_size    = risk_amount / (sl_pips * pip_value)
    
    # Non superare il 5% del capitale per trade
    max_lot = (equity * 0.05) / (sl_pips * pip_value)
    lot_size = min(lot_size, max_lot)
    
    return round(lot_size, 2)

def calculate_tp_sl(entry_price, direction, atr, pair):
    """
    Calcola Take Profit (TP) e Stop Loss (SL) usando l'ATR.
    direction: 'BUY' o 'SELL'
    """
    sl_distance = 1.5 * atr
    tp_distance = sl_distance * RR_RATIO
    
    if direction == "BUY":
        sl_price = round(entry_price - sl_distance, 5)
        tp_price = round(entry_price + tp_distance, 5)
    else:  # SELL
        sl_price = round(entry_price + sl_distance, 5)
        tp_price = round(entry_price - tp_distance, 5)
    
    # Converti SL in PIPS
    if "JPY" in pair:
        sl_pips = abs(sl_price - entry_price) * 100
    else:
        sl_pips = abs(sl_price - entry_price) * 10000
    
    return tp_price, sl_price, sl_pips
