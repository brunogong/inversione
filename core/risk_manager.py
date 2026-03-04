from dataclasses import dataclass
from config import settings


@dataclass
class PositionSize:
    lot_size: float
    risk_amount: float
    risk_pct: float
    margin_required: float
    pip_value: float


class RiskManager:
    STANDARD_LOT = 100_000

    def __init__(self, balance=None, risk_pct=None):
        self.balance = balance or settings.ACCOUNT_BALANCE
        self.risk_pct = risk_pct or settings.RISK_PER_TRADE
        self.max_daily_risk = settings.MAX_DAILY_RISK
        self.daily_risk_used = 0.0

    def calculate_position(self, pair, entry, stop_loss, direction):
        sl_dist = abs(entry - stop_loss)
        pip_sz = 0.01 if "JPY" in pair else 0.0001
        sl_pips = sl_dist / pip_sz
        if sl_pips == 0:
            return PositionSize(0, 0, 0, 0, 0)

        pip_val = self._pip_value(pair, entry)
        risk_amt = self.balance * (self.risk_pct / 100)
        remaining = self.balance * (self.max_daily_risk / 100) - self.daily_risk_used
        risk_amt = min(risk_amt, max(remaining, 0))
        if risk_amt <= 0:
            return PositionSize(0, 0, 0, 0, 0)

        lot_size = risk_amt / (sl_pips * pip_val)
        lot_size = max(0.01, round(lot_size, 2))
        actual_risk = lot_size * sl_pips * pip_val
        actual_pct = (actual_risk / self.balance) * 100
        margin = (lot_size * self.STANDARD_LOT * entry) / 100

        return PositionSize(lot_size, round(actual_risk, 2), round(actual_pct, 2), round(margin, 2), round(pip_val, 4))

    def _pip_value(self, pair, price):
        quote = pair[3:]
        if quote == "USD":
            return 10.0
        elif "JPY" in pair:
            return (10.0 / price) * 100 if price > 0 else 10.0
        else:
            return 10.0 / price if price > 0 else 10.0

    def register_trade(self, risk_amount):
        self.daily_risk_used += risk_amount

    def reset_daily(self):
        self.daily_risk_used = 0.0
