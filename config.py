# config.py

# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"     # Ottenuto da @BotFather
TELEGRAM_CHAT_ID   = "YOUR_TELEGRAM_CHAT_ID"        # Ottenuto da @userinfobot

# ===== ANTHROPIC (Claude AI) =====
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"      # Ottenuto da https://console.anthropic.com/

# ===== PARAMETRI DI TRADING =====
ACCOUNT_EQUITY    = 5000.0                       # Capitale disponibile (aggiorna qui!)
RISK_PERCENT      = 1.0                           # % del capitale a rischio per trade
RR_RATIO          = 2.0                           # Risk/Reward Ratio (1:2)
CURRENCY_PAIRS    = ["EURUSD", "GBPUSD", "USDJPY"] # Coppie da monitorare
TIMEFRAMES        = ["15m", "1h", "4h"]           # Timeframe analizzati

# ===== STRATEGIE =====
USE_CLAUDE_AI     = True                          # Abilitare conferma AI
LOOKBACK_PERIOD   = 50                            # Barre per analizzare la struttura
MIN_VOLUME_INC    = 1.5                           # Minimo aumento volume per confermare
