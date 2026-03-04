import numpy as np
import pandas as pd
import ta


class IndicatorEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate()

    def _validate(self):
        required = ["Open", "High", "Low", "Close"]
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"Colonna mancante: {c}")
        self.df = self.df.dropna(subset=required)
        if "Volume" not in self.df.columns:
            self.df["Volume"] = 0

    def compute_all(self) -> pd.DataFrame:
        df = self.df
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        for p in [8, 13, 21, 50, 100, 200]:
            df[f"EMA_{p}"] = ta.trend.ema_indicator(close, window=p)
        df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
        df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
        df["SMA_200"] = ta.trend.sma_indicator(close, window=200)

        df["RSI"] = ta.momentum.rsi(close, window=14)
        df["RSI_prev"] = df["RSI"].shift(1)

        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()
        df["MACD_hist_prev"] = df["MACD_hist"].shift(1)

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_mid"] = bb.bollinger_mavg()
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
        df["BB_pct"] = bb.bollinger_pband()

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["STOCH_K"] = stoch.stoch()
        df["STOCH_D"] = stoch.stoch_signal()

        adx_i = ta.trend.ADXIndicator(high, low, close, window=14)
        df["ADX"] = adx_i.adx()
        df["DI_plus"] = adx_i.adx_pos()
        df["DI_minus"] = adx_i.adx_neg()

        df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)

        ichi = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
        df["ICHI_tenkan"] = ichi.ichimoku_conversion_line()
        df["ICHI_kijun"] = ichi.ichimoku_base_line()
        df["ICHI_A"] = ichi.ichimoku_a()
        df["ICHI_B"] = ichi.ichimoku_b()

        df["CCI"] = ta.trend.cci(high, low, close, window=20)
        df["WILLR"] = ta.momentum.williams_r(high, low, close, lbp=14)

        df = self._market_structure(df, lookback=5)
        df = self._candle_patterns(df)
        df = self._support_resistance(df, window=20)
        df = self._rsi_divergence(df, lookback=14)
        df = self._order_blocks(df)
        df = self._fair_value_gaps(df)

        self.df = df
        return df

    def _market_structure(self, df, lookback=5):
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

    def _candle_patterns(self, df):
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

    def _support_resistance(self, df, window=20):
        df["Resistance"] = df["High"].rolling(window=window).max()
        df["Support"] = df["Low"].rolling(window=window).min()
        atr = df["ATR"] if "ATR" in df.columns else (df["High"] - df["Low"]).rolling(14).mean()
        threshold = atr * 0.5
        df["Near_Support"] = ((df["Close"] - df["Support"]).abs() < threshold).astype(int)
        df["Near_Resistance"] = ((df["Close"] - df["Resistance"]).abs() < threshold).astype(int)
        return df

    def _rsi_divergence(self, df, lookback=14):
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

    def _order_blocks(self, df):
        o, c, h, l = df["Open"].values, df["Close"].values, df["High"].values, df["Low"].values
        n = len(df)
        ob_bull, ob_bear = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(3, n):
            if c[i-1] < o[i-1] and c[i] > o[i] and c[i] > h[i-1] and (c[i]-o[i]) > (o[i-1]-c[i-1]) * 1.5:
                ob_bull[i] = l[i-1]
            if c[i-1] > o[i-1] and c[i] < o[i] and c[i] < l[i-1] and (o[i]-c[i]) > (c[i-1]-o[i-1]) * 1.5:
                ob_bear[i] = h[i-1]
        df["OB_Bull"] = ob_bull
        df["OB_Bear"] = ob_bear
        return df

    def _fair_value_gaps(self, df):
        h, l = df["High"].values, df["Low"].values
        n = len(df)
        fvg_b, fvg_s = np.zeros(n), np.zeros(n)
        for i in range(2, n):
            if l[i] > h[i-2]:
                fvg_b[i] = 1
            if h[i] < l[i-2]:
                fvg_s[i] = 1
        df["FVG_Bull"] = fvg_b
        df["FVG_Bear"] = fvg_s
        return df
