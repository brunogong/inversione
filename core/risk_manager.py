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

        # Usa la config centralizzata per pip size e pip value
        pip_sz = settings.get_pip_size(pair)
        sl_pips = sl_dist / pip_sz

        if sl_pips == 0:
            return PositionSize(0, 0, 0, 0, 0)

        pip_val = settings.get_pip_value(pair, entry)

        # Calcola il rischio
        risk_amt = self.balance * (self.risk_pct / 100)
        remaining = self.balance * (self.max_daily_risk / 100) - self.daily_risk_used
        risk_amt = min(risk_amt, max(remaining, 0))

        if risk_amt <= 0:
            return PositionSize(0, 0, 0, 0, 0)

        # Calcola lot size
        lot_size = risk_amt / (sl_pips * pip_val)

        # Lotti minimi diversi per metalli
        if pair in ["XAUUSD", "XAGUSD"]:
            lot_size = max(0.01, round(lot_size, 2))
        else:
            lot_size = max(0.01, round(lot_size, 2))

        actual_risk = lot_size * sl_pips * pip_val
        actual_pct = (actual_risk / self.balance) * 100

        # Margine (approssimato)
        if pair == "XAUUSD":
            margin = (lot_size * 100 * entry) / 100  # 100 oz per lotto
        elif pair == "XAGUSD":
            margin = (lot_size * 5000 * entry) / 100  # 5000 oz per lotto
        else:
            margin = (lot_size * self.STANDARD_LOT * entry) / 100

        return PositionSize(
            lot_size, round(actual_risk, 2),
            round(actual_pct, 2), round(margin, 2),
            round(pip_val, 4),
        )

    def register_trade(self, risk_amount):
        self.daily_risk_used += risk_amount

    def reset_daily(self):
        self.daily_risk_used = 0.0
