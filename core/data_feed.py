import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from config import settings


class DataFeed:

    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self.cache_ttl = timedelta(minutes=2)

    def get_candles(self, pair, interval="1h", period="60d"):
        cache_key = f"{pair}_{interval}_{period}"
        now = datetime.now()
        if cache_key in self._cache and cache_key in self._cache_time:
            if now - self._cache_time[cache_key] < self.cache_ttl:
                return self._cache[cache_key]

        # Usa il ticker corretto per ogni coppia
        ticker = settings.get_ticker(pair)

        try:
            data = yf.download(
                ticker, period=period, interval=interval,
                progress=False, auto_adjust=True,
            )
            if data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.rename(columns={
                "open": "Open", "high": "High",
                "low": "Low", "close": "Close",
                "volume": "Volume",
            })
            self._cache[cache_key] = data
            self._cache_time[cache_key] = now
            return data
        except Exception:
            return None

    def get_multi_timeframe(self, pair, timeframes):
        results = {}
        for tf_name, tf_cfg in timeframes.items():
            interval = tf_cfg["yf_interval"]
            period = tf_cfg["yf_period"]
            label = tf_cfg["label"]
            if interval == "4h":
                df_1h = self.get_candles(pair, "1h", "120d")
                if df_1h is not None and not df_1h.empty:
                    df = df_1h.resample("4h").agg({
                        "Open": "first", "High": "max",
                        "Low": "min", "Close": "last",
                        "Volume": "sum",
                    }).dropna()
                    results[label] = df
            else:
                df = self.get_candles(pair, interval, period)
                if df is not None:
                    results[label] = df
        return results

    def clear_cache(self):
        self._cache.clear()
        self._cache_time.clear()
