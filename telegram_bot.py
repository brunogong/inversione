# telegram_bot.py
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True
    }
    response = requests.post(url, data=data).json()
    return response

def format_signal_message(signal_data):
    """Formatta il messaggio per Telegram (MarkdownV2)"""
    escape = lambda s: s.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]')
    
    pair     = escape(signal_data['pair'])
    direction= escape(signal_data['direction'])
    entry    = escape(f"{signal_data['entry']:.5f}")
    tp       = escape(f"{signal_data['tp']:.5f}")
    sl       = escape(f"{signal_data['sl']:.5f}")
    lot      = escape(f"{signal_data['lot']:.2f}")
    timeframe= escape(signal_data['timeframe'])
    atr      = escape(f"{signal_data['atr']:.5f}")
    
    msg = f"""
📈 **NUOVO SEGNALE {pair}** 📉

🕒 *Timeframe*: {timeframe}
📍 *Segnalo*: **{direction}**
💰 *Entry*: `{entry}`
🎯 *Take Profit (TP)*: `{tp}`
🛑 *Stop Loss (SL)*: `{sl}`
📊 *Lotto*: **{lot}**
🌀 *ATR*: `{atr}`

📉 *Rischio*: {signal_data['sl_pips']:.1f} pips \\= {RISK_PERCENT}% del capitale

✅ **Confermato da AI Claude!**

⚠️ **Esegui solo dopo aver verificato manualmente!**
"""
    return msg
