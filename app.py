import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
import json

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

from config import settings
from core.data_feed import DataFeed
from core.indicators import IndicatorEngine
from core.strategy import StrategyEngine, Signal
from core.risk_manager import RiskManager
from core.notifier import TelegramNotifier

st.set_page_config(page_title="Forex Reversal Engine", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0a0e17; }
.main-header { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); padding: 1.5rem 2rem; border-radius: 16px; border: 1px solid #1e3a5f; margin-bottom: 1.5rem; text-align: center; }
.main-header h1 { background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.2rem; font-weight: 800; margin: 0; }
.main-header p { color: #64748b; font-size: 0.95rem; margin-top: 6px; }
.signal-card { background: #1a2332; border: 1px solid #1e3a5f; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #3b82f6; }
.signal-card.buy { border-left-color: #10b981; }
.signal-card.sell { border-left-color: #ef4444; }
.signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.pair-name { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; }
.dir-badge { padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
.dir-badge.buy { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.dir-badge.sell { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.score-big { font-size: 1.8rem; font-weight: 700; color: #f59e0b; }
.score-max { color: #64748b; font-size: 0.9rem; }
.trade-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 1rem 0; }
.trade-item { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 10px 12px; }
.trade-item .t-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; }
.trade-item .t-value { font-size: 1rem; font-weight: 600; font-family: monospace; color: #e2e8f0; }
.trade-item .t-value.green { color: #10b981; }
.trade-item .t-value.red { color: #ef4444; }
.trade-item .t-value.gold { color: #f59e0b; }
.ind-tag { display: inline-block; font-size: 0.75rem; padding: 4px 10px; border-radius: 6px; margin: 3px; }
.ind-tag.bull { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25); color: #10b981; }
.ind-tag.bear { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25); color: #ef4444; }
.strength-badge { padding: 4px 14px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; background: rgba(245,158,11,0.15); color: #f59e0b; display: inline-block; }
.meta-row { display: flex; justify-content: space-between; align-items: center; margin-top: 12px; font-size: 0.8rem; color: #64748b; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; }
.live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5); animation: pulse 2s infinite; margin-right: 6px; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
#MainMenu {visibility:hidden} footer {visibility:hidden} .stDeployButton {display:none}
div[data-testid="stMetric"] { background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

defaults = {
    "signals": [], "scan_count": 0, "signal_count": 0,
    "last_scan": "Never", "scanning": False, "cooldowns": {},
    "log": [], "balance": settings.ACCOUNT_BALANCE,
    "risk_pct": settings.RISK_PER_TRADE, "min_confluence": settings.MIN_CONFLUENCE,
    "selected_pairs": settings.PAIRS.copy(), "auto_scan": False, "auto_telegram": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.insert(0, f"[{ts}] {msg}")
    st.session_state.log = st.session_state.log[:100]


@st.cache_resource
def get_data_feed():
    return DataFeed()


@st.cache_resource
def get_notifier():
    return TelegramNotifier()


def signal_to_dict(s):
    return {
        "id": s.id, "pair": s.pair, "direction": s.direction,
        "entry": s.entry, "stop_loss": s.stop_loss,
        "take_profit_1": s.take_profit_1, "take_profit_2": s.take_profit_2,
        "take_profit_3": s.take_profit_3, "lot_size": s.lot_size,
        "risk_amount": s.risk_amount, "risk_pct": s.risk_pct,
        "rr_ratio": s.rr_ratio, "sl_pips": s.sl_pips, "tp1_pips": s.tp1_pips,
        "confluence_score": s.confluence_score, "max_score": s.max_score,
        "timeframe": s.timeframe, "indicators": s.indicators,
        "timestamp": s.timestamp, "session": s.session, "strength": s.strength,
    }


def render_signal_card(s):
    d = s["direction"]
    dc = "buy" if d == "BUY" else "sell"
    dir_emoji = "🟢" if d == "BUY" else "🔴"
    dir_text = f"{dir_emoji} {d} {'LONG' if d == 'BUY' else 'SHORT'}"
    indicators_html = ""
    for key, info in s.get("indicators", {}).items():
        cls = "bull" if info["signal"] == "BULL" else "bear"
        indicators_html += f'<span class="ind-tag {cls}">{info["detail"]}</span>'

    st.markdown(f"""
    <div class="signal-card {dc}">
        <div class="signal-header">
            <div><span class="pair-name">{s['pair']}</span>
            <span class="dir-badge {dc}" style="margin-left:12px">{dir_text}</span></div>
            <div style="text-align:right"><div class="score-big">{s['confluence_score']}</div>
            <div class="score-max">/ {s['max_score']}</div></div>
        </div>
        <div class="trade-grid">
            <div class="trade-item"><div class="t-label">Entry</div><div class="t-value">{s['entry']}</div></div>
            <div class="trade-item"><div class="t-label">Stop Loss</div><div class="t-value red">{s['stop_loss']} ({s['sl_pips']}p)</div></div>
            <div class="trade-item"><div class="t-label">TP1</div><div class="t-value green">{s['take_profit_1']} ({s['tp1_pips']}p)</div></div>
            <div class="trade-item"><div class="t-label">TP2</div><div class="t-value green">{s['take_profit_2']}</div></div>
            <div class="trade-item"><div class="t-label">TP3</div><div class="t-value green">{s['take_profit_3']}</div></div>
            <div class="trade-item"><div class="t-label">Lots</div><div class="t-value gold">{s['lot_size']}</div></div>
            <div class="trade-item"><div class="t-label">Risk</div><div class="t-value">${s['risk_amount']} ({s['risk_pct']}%)</div></div>
            <div class="trade-item"><div class="t-label">R:R</div><div class="t-value">1:{s['rr_ratio']}</div></div>
        </div>
        <div style="margin-top:10px"><div class="t-label" style="margin-bottom:6px">INDICATORS ({len(s.get('indicators',{}))})</div>{indicators_html}</div>
        <div class="meta-row"><span>{s['timestamp']} · {s['timeframe']} · {s.get('session','').upper()}</span>
        <span class="strength-badge">{s['strength']}</span></div>
    </div>""", unsafe_allow_html=True)


def build_chart(df, pair, signal_dict=None):
    df_plot = df.tail(100).copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2], subplot_titles=[f"{pair} — H1", "RSI", "MACD"])
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"], increasing_line_color="#10b981", decreasing_line_color="#ef4444", name="Price"), row=1, col=1)
    for ema, color in [("EMA_21", "#60a5fa"), ("EMA_50", "#f59e0b"), ("EMA_200", "#a78bfa")]:
        if ema in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[ema], line=dict(width=1, color=color), name=ema, opacity=0.7), row=1, col=1)
    if signal_dict:
        fig.add_hline(y=signal_dict["entry"], line_color="#3b82f6", line_dash="dash", annotation_text=f"Entry {signal_dict['entry']}", row=1, col=1)
        fig.add_hline(y=signal_dict["stop_loss"], line_color="#ef4444", line_dash="dash", annotation_text=f"SL {signal_dict['stop_loss']}", row=1, col=1)
        fig.add_hline(y=signal_dict["take_profit_1"], line_color="#10b981", line_dash="dash", annotation_text=f"TP1 {signal_dict['take_profit_1']}", row=1, col=1)
    if "RSI" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["RSI"], line=dict(width=1.5, color="#a78bfa"), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="#ef4444", line_dash="dot", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_color="#10b981", line_dash="dot", opacity=0.5, row=2, col=1)
    if "MACD" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD"], line=dict(width=1.5, color="#3b82f6"), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD_signal"], line=dict(width=1, color="#f59e0b"), name="Signal"), row=3, col=1)
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in df_plot["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["MACD_hist"], marker_color=colors, name="Hist", opacity=0.5), row=3, col=1)
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17", height=700, margin=dict(l=50, r=20, t=40, b=20), xaxis_rangeslider_visible=False, showlegend=False, font=dict(color="#e2e8f0"))
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def run_full_scan():
    st.session_state.scanning = True
    add_log("═══ SCAN STARTED ═══")
    feed = get_data_feed()
    strategy = StrategyEngine(weights=settings.WEIGHTS)
    risk_mgr = RiskManager(balance=st.session_state.balance, risk_pct=st.session_state.risk_pct)
    notifier = get_notifier()
    new_signals = []
    progress = st.progress(0, text="Scanning market...")
    status = st.empty()
    now = datetime.now(timezone.utc)
    pairs = st.session_state.selected_pairs
    total = len(pairs)

    for idx, pair in enumerate(pairs):
        progress.progress((idx + 1) / total, text=f"Scanning {pair}...")
        status.markdown(f"**Analyzing {pair}** — {idx+1}/{total}")
        try:
            cd = st.session_state.cooldowns.get(pair)
            if cd and (now - cd) < timedelta(minutes=settings.COOLDOWN_MINUTES):
                continue
            tf_data = feed.get_multi_timeframe(pair, settings.TIMEFRAMES)
            primary_df = tf_data.get("H1")
            if primary_df is None or len(primary_df) < 200:
                add_log(f"  {pair}: insufficient data")
                continue
            engine = IndicatorEngine(primary_df)
            primary_df = engine.compute_all()
            htf_bias = None
            for htf_label in ["H4", "D1"]:
                htf_df = tf_data.get(htf_label)
                if htf_df is not None and len(htf_df) > 50:
                    htf_eng = IndicatorEngine(htf_df)
                    htf_df = htf_eng.compute_all()
                    last_htf = htf_df.iloc[-1]
                    e50 = last_htf.get("EMA_50", np.nan)
                    e200 = last_htf.get("EMA_200", np.nan)
                    if not np.isnan(e50) and not np.isnan(e200):
                        htf_bias = 1 if e50 > e200 else -1
                    break
            signal = strategy.analyze(df=primary_df, pair=pair, timeframe="H1", htf_bias=htf_bias)
            if signal and signal.confluence_score >= st.session_state.min_confluence:
                pos = risk_mgr.calculate_position(pair=pair, entry=signal.entry, stop_loss=signal.stop_loss, direction=signal.direction)
                signal.lot_size = pos.lot_size
                signal.risk_amount = pos.risk_amount
                signal.risk_pct = pos.risk_pct
                if pos.lot_size > 0:
                    new_signals.append((signal, primary_df))
                    st.session_state.cooldowns[pair] = now
                    risk_mgr.register_trade(pos.risk_amount)
                    add_log(f"  🎯 {pair} {signal.direction} | Score: {signal.confluence_score} | Lots: {signal.lot_size}")
            else:
                add_log(f"  {pair}: no signal")
        except Exception as e:
            add_log(f"  ❌ {pair}: {str(e)[:60]}")

    progress.empty()
    status.empty()

    for signal, df_data in new_signals:
        sig_dict = signal_to_dict(signal)
        st.session_state.signals.insert(0, sig_dict)
        st.session_state.signal_count += 1
        if st.session_state.auto_telegram:
            try:
                sent = notifier.send_signal(signal)
                add_log(f"  📤 Telegram {'sent' if sent else 'failed'}: {signal.pair}")
            except Exception as e:
                add_log(f"  ❌ Telegram error: {e}")

    st.session_state.signals = st.session_state.signals[:50]
    st.session_state.scan_count += 1
    st.session_state.last_scan = now.strftime("%Y-%m-%d %H:%M UTC")
    st.session_state.scanning = False
    add_log(f"═══ SCAN COMPLETE — {len(new_signals)} signals ═══")
    return new_signals


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.session_state.balance = st.number_input("💰 Balance ($)", min_value=100.0, max_value=10000000.0, value=st.session_state.balance, step=500.0)
    st.session_state.risk_pct = st.slider("📐 Risk %", 0.5, 5.0, st.session_state.risk_pct, 0.25)
    st.session_state.min_confluence = st.slider("🎯 Min Confluence", 3.0, 15.0, st.session_state.min_confluence, 0.5)
    st.markdown("---")
    st.markdown("## 💱 Pairs")
    st.session_state.selected_pairs = st.multiselect("Select pairs", settings.PAIRS, default=st.session_state.selected_pairs)
    st.markdown("---")
    st.markdown("## 📡 Telegram")
    st.session_state.auto_telegram = st.toggle("Auto-send", value=st.session_state.auto_telegram)
    tg_ok = "✅" if settings.TELEGRAM_TOKEN else "❌"
    st.caption(f"Status: {tg_ok}")
    if settings.TELEGRAM_TOKEN and st.button("📤 Test"):
        ok = get_notifier().send_message("🧪 <b>Test</b> — Engine connected!")
        st.success("Sent!" if ok else "Failed!")
    st.markdown("---")
    st.markdown("## 📋 Log")
    log_c = st.container(height=250)
    with log_c:
        for entry in st.session_state.log[:30]:
            st.caption(entry)

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
st.markdown("""<div class="main-header"><h1>⚡ Forex Reversal Engine</h1><p>Institutional-Grade Multi-Timeframe Trend Reversal Scanner</p></div>""", unsafe_allow_html=True)

if HAS_AUTOREFRESH and st.session_state.auto_scan:
    st_autorefresh(interval=settings.SCAN_INTERVAL * 60 * 1000, limit=None, key="auto_refresh")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    scan_btn = st.button("🔍 **SCAN MARKET NOW**", use_container_width=True, type="primary")
with c2:
    st.session_state.auto_scan = st.toggle("⏰ Auto-scan", st.session_state.auto_scan)
with c3:
    if st.button("🗑️ Clear"):
        st.session_state.signals = []
        st.session_state.cooldowns = {}
        st.rerun()
with c4:
    if st.button("🔄 Cache"):
        get_data_feed().clear_cache()
        st.rerun()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📊 Scans", st.session_state.scan_count)
m2.metric("🎯 Signals", st.session_state.signal_count)
m3.metric("📡 Active", len(st.session_state.signals))
m4.metric("💱 Pairs", len(st.session_state.selected_pairs))
m5.metric("⏱ Last Scan", st.session_state.last_scan)

st.markdown("---")

if scan_btn:
    new = run_full_scan()
    st.toast(f"🎯 {len(new)} signal(s)!" if new else "No signals", icon="🎯" if new else "📊")
    st.rerun()

if st.session_state.auto_scan and HAS_AUTOREFRESH and st.session_state.scan_count == 0:
    run_full_scan()
    st.rerun()

if st.session_state.signals:
    st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem"><span class="live-dot"></span><span style="font-size:1.2rem;font-weight:600;color:#e2e8f0">Live Signal Feed — {len(st.session_state.signals)} signals</span></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📡 Signals", "📈 Charts", "📊 Analysis"])
    with tab1:
        for sig in st.session_state.signals:
            render_signal_card(sig)
    with tab2:
        if st.session_state.signals:
            sel = st.selectbox("Pair", [s["pair"] for s in st.session_state.signals])
            sig_c = next((s for s in st.session_state.signals if s["pair"] == sel), None)
            if sig_c:
                with st.spinner(f"Loading {sel}..."):
                    cdf = get_data_feed().get_candles(sel, "1h", "60d")
                    if cdf is not None:
                        cdf = IndicatorEngine(cdf).compute_all()
                        st.plotly_chart(build_chart(cdf, sel, sig_c), use_container_width=True)
    with tab3:
        df_t = pd.DataFrame([{"Pair": s["pair"], "Dir": s["direction"], "Entry": s["entry"], "SL": s["stop_loss"], "TP1": s["take_profit_1"], "Lots": s["lot_size"], "Risk$": s["risk_amount"], "R:R": f"1:{s['rr_ratio']}", "Score": f"{s['confluence_score']}/{s['max_score']}", "Strength": s["strength"]} for s in st.session_state.signals])
        st.dataframe(df_t, use_container_width=True, hide_index=True)
        st.download_button("📥 Export JSON", json.dumps(st.session_state.signals, indent=2, default=str), f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json")
else:
    st.markdown("""<div style="text-align:center;padding:60px;color:#64748b"><div style="font-size:4rem">📡</div><h2 style="color:#e2e8f0">Waiting for signals...</h2><p>Click <b>SCAN MARKET NOW</b> or enable <b>Auto-scan</b></p></div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#475569;font-size:0.8rem;padding:1rem">⚡ Forex Reversal Engine v2.0<br><span style="color:#ef4444">⚠️ Trading involves risk.</span></div>', unsafe_allow_html=True)
