import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict
from config import settings


@dataclass
class Signal:
    pair: str
    direction: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    lot_size: float
    risk_amount: float
    risk_pct: float
    rr_ratio: float
    sl_pips: float
    tp1_pips: float
    confluence_score: float
    max_score: float
    timeframe: str
    indicators: Dict = field(default_factory=dict)
    timestamp: str = ""
    session: str = ""
    strength: str = ""
    id: str = ""


class StrategyEngine:

    def __init__(self, weights=None):
        self.weights = weights or settings.WEIGHTS

    def analyze(self, df, pair, timeframe, htf_bias=None):
        if len(df) < 200:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        scores = {}
        details = {}

        def _g(name, default=np.nan):
            v = last.get(name, default)
            if isinstance(v, float) and np.isnan(v):
                return default
            return v

        def _gp(name, default=np.nan):
            v = prev.get(name, default)
            if isinstance(v, float) and np.isnan(v):
                return default
            return v

        close = _g("Close")

        e50, e200 = _g("EMA_50"), _g("EMA_200")
        if e50 is not np.nan and e200 is not np.nan:
            scores["ema_trend"] = 1 if e50 > e200 else -1
            details["ema_trend"] = f"EMA 50 {'>' if e50 > e200 else '<'} 200"

        e8, e21, pe8, pe21 = _g("EMA_8"), _g("EMA_21"), _gp("EMA_8"), _gp("EMA_21")
        if not any(v is np.nan for v in [e8, e21, pe8, pe21]):
            if pe8 <= pe21 and e8 > e21:
                scores["ema_cross"] = 1
                details["ema_cross"] = "EMA 8/21 Bullish cross ↑"
            elif pe8 >= pe21 and e8 < e21:
                scores["ema_cross"] = -1
                details["ema_cross"] = "EMA 8/21 Bearish cross ↓"
            else:
                scores["ema_cross"] = 0

        rsi = _g("RSI", 50)
        rsi_p = _gp("RSI", 50)
        if rsi < 30:
            scores["rsi_reversal"] = 1
            details["rsi_reversal"] = f"RSI {rsi:.1f} Oversold"
        elif rsi > 70:
            scores["rsi_reversal"] = -1
            details["rsi_reversal"] = f"RSI {rsi:.1f} Overbought"
        elif rsi_p < 30 and rsi > 30:
            scores["rsi_reversal"] = 1
            details["rsi_reversal"] = "RSI leaving oversold ↑"
        elif rsi_p > 70 and rsi < 70:
            scores["rsi_reversal"] = -1
            details["rsi_reversal"] = "RSI leaving overbought ↓"
        else:
            scores["rsi_reversal"] = 0

        rd = _g("RSI_Divergence", 0)
        if rd != 0:
            scores["rsi_divergence"] = int(rd)
            details["rsi_divergence"] = "Bullish RSI divergence 🟢" if rd > 0 else "Bearish RSI divergence 🔴"
        else:
            scores["rsi_divergence"] = 0

        m, ms, pm, pms = _g("MACD", 0), _g("MACD_signal", 0), _gp("MACD", 0), _gp("MACD_signal", 0)
        if not any(v is np.nan for v in [m, ms, pm, pms]):
            if pm <= pms and m > ms:
                scores["macd_cross"] = 1
                details["macd_cross"] = "MACD Bullish cross ↑"
            elif pm >= pms and m < ms:
                scores["macd_cross"] = -1
                details["macd_cross"] = "MACD Bearish cross ↓"
            else:
                scores["macd_cross"] = 0

        mh, mhp = _g("MACD_hist", 0), _g("MACD_hist_prev", 0)
        if mhp < 0 < mh:
            scores["macd_histogram"] = 1
            details["macd_histogram"] = "MACD Hist turning bullish"
        elif mhp > 0 > mh:
            scores["macd_histogram"] = -1
            details["macd_histogram"] = "MACD Hist turning bearish"
        else:
            scores["macd_histogram"] = 0

        bbp = _g("BB_pct", 0.5)
        if bbp < 0.05:
            scores["bollinger"] = 1
            details["bollinger"] = "Lower Bollinger — reversal"
        elif bbp > 0.95:
            scores["bollinger"] = -1
            details["bollinger"] = "Upper Bollinger — reversal"
        else:
            scores["bollinger"] = 0

        sk, sd, psk, psd = _g("STOCH_K", 50), _g("STOCH_D", 50), _gp("STOCH_K", 50), _gp("STOCH_D", 50)
        if sk < 20 and psk <= psd and sk > sd:
            scores["stochastic"] = 1
            details["stochastic"] = f"Stoch oversold cross ({sk:.0f})"
        elif sk > 80 and psk >= psd and sk < sd:
            scores["stochastic"] = -1
            details["stochastic"] = f"Stoch overbought cross ({sk:.0f})"
        else:
            scores["stochastic"] = 0

        adx = _g("ADX", 0)
        dip, dim = _g("DI_plus", 0), _g("DI_minus", 0)
        if adx > 20:
            scores["adx_strength"] = 1 if dip > dim else -1
            details["adx_strength"] = f"ADX {adx:.0f} {'Bull' if dip > dim else 'Bear'}"
        else:
            scores["adx_strength"] = 0

        ia, ib, tk, kj = _g("ICHI_A"), _g("ICHI_B"), _g("ICHI_tenkan"), _g("ICHI_kijun")
        if not any(v is np.nan for v in [ia, ib, tk, kj]):
            ct, cb = max(ia, ib), min(ia, ib)
            if close > ct and tk > kj:
                scores["ichimoku"] = 1
                details["ichimoku"] = "Above Ichimoku cloud"
            elif close < cb and tk < kj:
                scores["ichimoku"] = -1
                details["ichimoku"] = "Below Ichimoku cloud"
            else:
                scores["ichimoku"] = 0

        sb = _g("Structure_Break", 0)
        st_val = _g("Structure", 0)
        if sb != 0:
            scores["market_structure"] = int(sb)
            details["market_structure"] = f"⚡ CHoCH {'Bullish' if sb > 0 else 'Bearish'}"
        elif st_val != 0:
            scores["market_structure"] = int(st_val) * 0.5
            details["market_structure"] = "Higher H/L" if st_val > 0 else "Lower H/L"
        else:
            scores["market_structure"] = 0

        cp = _g("Candle_Pattern", 0)
        if cp != 0:
            scores["candle_pattern"] = int(cp)
            details["candle_pattern"] = f"{'Bullish' if cp > 0 else 'Bearish'} reversal candle 🕯"
        else:
            scores["candle_pattern"] = 0

        ns, nr = _g("Near_Support", 0), _g("Near_Resistance", 0)
        if ns:
            scores["support_resistance"] = 1
            details["support_resistance"] = "Near Support"
        elif nr:
            scores["support_resistance"] = -1
            details["support_resistance"] = "Near Resistance"
        else:
            scores["support_resistance"] = 0

        vol = _g("Volume", 0)
        if vol > 0:
            avg_vol = df["Volume"].tail(20).mean()
            if avg_vol > 0 and vol > avg_vol * 1.5:
                scores["volume_confirm"] = 1 if close > last["Open"] else -1
                details["volume_confirm"] = "High volume confirmation"
            else:
                scores["volume_confirm"] = 0
        else:
            scores["volume_confirm"] = 0

        bull_score = bear_score = max_possible = 0.0
        for key, value in scores.items():
            w = self.weights.get(key, 1.0)
            max_possible += w
            if value > 0:
                bull_score += value * w
            elif value < 0:
                bear_score += abs(value) * w

        if bull_score > bear_score:
            direction, confluence = "BUY", bull_score
        elif bear_score > bull_score:
            direction, confluence = "SELL", bear_score
        else:
            return None

        if htf_bias is not None:
            if (direction == "BUY" and htf_bias < 0) or (direction == "SELL" and htf_bias > 0):
                confluence *= 0.6

        if confluence < settings.MIN_CONFLUENCE:
            return None

        now = datetime.now(timezone.utc)
        session = self._session(now.hour)
        if session == "off_hours":
            confluence *= 0.8
            if confluence < settings.MIN_CONFLUENCE:
                return None

        active = {}
        for key, val in scores.items():
            if val != 0 and key in details:
                active[key] = {
                    "signal": "BULL" if val > 0 else "BEAR",
                    "detail": details[key],
                    "weight": self.weights.get(key, 1.0),
                    "contribution": round(abs(val) * self.weights.get(key, 1.0), 2),
                }

        pct = confluence / max_possible * 100 if max_possible > 0 else 0
        if pct > 70:
            strength = "🔥 VERY STRONG"
        elif pct > 55:
            strength = "💪 STRONG"
        elif pct > 40:
            strength = "📊 MODERATE"
        else:
            strength = "⚠️ WEAK"

        atr = _g("ATR", 0)
        if atr == 0 or atr is np.nan:
            return None

        entry = close
        if direction == "BUY":
            sl = entry - atr * 1.5
            tp1, tp2, tp3 = entry + atr * 2.0, entry + atr * 3.0, entry + atr * 5.0
        else:
            sl = entry + atr * 1.5
            tp1, tp2, tp3 = entry - atr * 2.0, entry - atr * 3.0, entry - atr * 5.0

        sl_dist = abs(entry - sl)
        tp1_dist = abs(tp1 - entry)
        rr = tp1_dist / sl_dist if sl_dist > 0 else 0
        if rr < settings.MIN_RR_RATIO:
            return None

        pip_sz = 0.01 if "JPY" in pair else 0.0001

        return Signal(
            pair=pair, direction=direction,
            entry=round(entry, 5), stop_loss=round(sl, 5),
            take_profit_1=round(tp1, 5), take_profit_2=round(tp2, 5),
            take_profit_3=round(tp3, 5),
            lot_size=0, risk_amount=0, risk_pct=settings.RISK_PER_TRADE,
            rr_ratio=round(rr, 2),
            sl_pips=round(sl_dist / pip_sz, 1),
            tp1_pips=round(tp1_dist / pip_sz, 1),
            confluence_score=round(confluence, 2),
            max_score=round(max_possible, 2),
            timeframe=timeframe, indicators=active,
            timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
            session=session, strength=strength,
            id=f"{pair}_{direction}_{now.strftime('%Y%m%d%H%M')}",
        )

    def _session(self, h):
        for name, t in settings.SESSIONS.items():
            if t["start"] <= h < t["end"]:
                return name
        return "off_hours"
