import httpx
from config import settings


class TelegramNotifier:
    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, token=None, chat_id=None):
        self.token = token or settings.TELEGRAM_TOKEN
        self.chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        self.url = self.BASE_URL.format(token=self.token)

    def send_signal(self, signal):
        return self._send(self._format(signal))

    def send_message(self, text):
        return self._send(text)

    def _send(self, text):
        if not self.token or not self.chat_id:
            return False
        try:
            with httpx.Client(timeout=10) as client:
                r = client.post(f"{self.url}/sendMessage", json={
                    "chat_id": self.chat_id, "text": text,
                    "parse_mode": "HTML", "disable_web_page_preview": True,
                })
                return r.status_code == 200
        except Exception:
            return False

    def _format(self, s):
        emoji = "🟢 BUY LONG" if s.direction == "BUY" else "🔴 SELL SHORT"
        fire = "🔥" if "STRONG" in s.strength else "📊"
        ind_lines = []
        for key, info in s.indicators.items():
            icon = "✅" if ((s.direction == "BUY" and info["signal"] == "BULL") or (s.direction == "SELL" and info["signal"] == "BEAR")) else "⬜"
            ind_lines.append(f"  {icon} {info['detail']}")
        indicators_text = "\n".join(ind_lines) or "  N/A"

        return f"""{fire} <b>FOREX SIGNAL — {s.strength}</b> {fire}
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 <b>{s.pair}</b> | <b>{emoji}</b>
⏰ {s.timeframe} | Session: {s.session.upper()}
📈 Confluence: <b>{s.confluence_score}/{s.max_score}</b>

💰 <b>TRADE SETUP:</b>
  ▫️ Entry:  <code>{s.entry}</code>
  ▫️ SL:     <code>{s.stop_loss}</code>  ({s.sl_pips} pips)
  ▫️ TP1:    <code>{s.take_profit_1}</code>  ({s.tp1_pips} pips)
  ▫️ TP2:    <code>{s.take_profit_2}</code>
  ▫️ TP3:    <code>{s.take_profit_3}</code>

📐 <b>RISK:</b>
  ▫️ Lots: <b>{s.lot_size}</b>
  ▫️ Risk: ${s.risk_amount} ({s.risk_pct}%)
  ▫️ R:R:  1:{s.rr_ratio}

📊 <b>INDICATORS ({len(s.indicators)}):</b>
{indicators_text}

⏱ {s.timestamp}
━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 <i>Forex Reversal Engine v2.0</i>"""
