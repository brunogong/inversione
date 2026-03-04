from dataclasses import dataclass, field
from typing import List, Dict


def _secret(key, default):
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


@dataclass
class Settings:
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    ACCOUNT_BALANCE: float = 10000.0
    ACCOUNT_CURRENCY: str = "USD"
    RISK_PER_TRADE: float = 1.5
    MAX_DAILY_RISK: float = 5.0
    MAX_OPEN_TRADES: int = 5
    MIN_CONFLUENCE: float = 6.0
    MIN_RR_RATIO: float = 2.0
    COOLDOWN_MINUTES: int = 60
    SCAN_INTERVAL: int = 5

    PAIRS: List[str] = field(default_factory=lambda: [
        # ── Metalli ────────────────────────
        "XAUUSD",   # Oro
        "XAGUSD",   # Argento
        # ── Major ─────────────────────────
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "USDCHF",
        "AUDUSD",
        "NZDUSD",
        "USDCAD",
        # ── Cross EUR ─────────────────────
        "EURJPY",
        "EURGBP",
        "EURAUD",
        "EURCHF",
        "EURNZD",
        "EURCAD",
        # ── Cross GBP ─────────────────────
        "GBPJPY",
        "GBPCHF",
        "GBPAUD",
        "GBPNZD",
        "GBPCAD",
        # ── Cross AUD/NZD ─────────────────
        "AUDNZD",
        "AUDCAD",
        "AUDJPY",
        "NZDJPY",
        "NZDCAD",
        # ── Cross JPY/CHF ─────────────────
        "CADJPY",
        "CHFJPY",
    ])

    # Ticker speciali per yfinance (quelli che non seguono il formato XXXYYY=X)
    SPECIAL_TICKERS: Dict = field(default_factory=lambda: {
        "XAUUSD": "XAUUSD=X",
        "XAGUSD": "XAGUSD=X",
    })

    # Configurazione pip per asset speciali
    PIP_CONFIG: Dict = field(default_factory=lambda: {
        "XAUUSD": {"pip_size": 0.01,  "pip_value_lot": 1.0,   "name": "Gold"},
        "XAGUSD": {"pip_size": 0.001, "pip_value_lot": 1.0,   "name": "Silver"},
        # Le coppie JPY hanno pip_size 0.01 (gestito automaticamente)
        # Tutte le altre hanno pip_size 0.0001 (default)
    })

    TIMEFRAMES: Dict = field(default_factory=lambda: {
        "primary": {"yf_interval": "1h", "yf_period": "60d", "label": "H1"},
        "confirm": {"yf_interval": "4h", "yf_period": "120d", "label": "H4"},
        "higher":  {"yf_interval": "1d", "yf_period": "365d", "label": "D1"},
    })

    WEIGHTS: Dict = field(default_factory=lambda: {
        "ema_trend": 2.0, "ema_cross": 1.5, "rsi_reversal": 1.0,
        "rsi_divergence": 2.0, "macd_cross": 1.5, "macd_histogram": 1.0,
        "bollinger": 1.0, "stochastic": 1.0, "adx_strength": 0.5,
        "ichimoku": 1.5, "market_structure": 2.5, "candle_pattern": 1.0,
        "support_resistance": 1.5, "volume_confirm": 0.5,
    })

    SESSIONS: Dict = field(default_factory=lambda: {
        "tokyo": {"start": 0, "end": 9},
        "london": {"start": 7, "end": 16},
        "newyork": {"start": 12, "end": 21},
    })

    def __post_init__(self):
        self.TELEGRAM_TOKEN = str(_secret("TELEGRAM_BOT_TOKEN", self.TELEGRAM_TOKEN))
        self.TELEGRAM_CHAT_ID = str(_secret("TELEGRAM_CHAT_ID", self.TELEGRAM_CHAT_ID))
        self.ACCOUNT_BALANCE = float(_secret("ACCOUNT_BALANCE", self.ACCOUNT_BALANCE))
        self.RISK_PER_TRADE = float(_secret("RISK_PER_TRADE", self.RISK_PER_TRADE))
        self.MIN_CONFLUENCE = float(_secret("MIN_CONFLUENCE", self.MIN_CONFLUENCE))
        self.SCAN_INTERVAL = int(_secret("SCAN_INTERVAL_MINUTES", self.SCAN_INTERVAL))

    def get_pip_size(self, pair):
        """Restituisce la dimensione del pip per una coppia."""
        if pair in self.PIP_CONFIG:
            return self.PIP_CONFIG[pair]["pip_size"]
        elif "JPY" in pair:
            return 0.01
        else:
            return 0.0001

    def get_pip_value(self, pair, price):
        """Restituisce il valore del pip per 1 lotto standard."""
        if pair in self.PIP_CONFIG:
            cfg = self.PIP_CONFIG[pair]
            return cfg["pip_value_lot"]

        quote = pair[3:6]
        if quote == "USD":
            return 10.0
        elif "JPY" in pair:
            return (10.0 / price) * 100 if price > 0 else 10.0
        else:
            return 10.0 / price if price > 0 else 10.0

    def get_ticker(self, pair):
        """Restituisce il ticker yfinance per una coppia."""
        if pair in self.SPECIAL_TICKERS:
            return self.SPECIAL_TICKERS[pair]
        return f"{pair[:3]}{pair[3:]}=X"


settings = Settings()
