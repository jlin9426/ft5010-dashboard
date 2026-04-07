"""
FT5010 Final Project - Strategy Dashboard
Group 6

Strategy: Hybrid Trend-Following + Mean-Reversion on EUR/USD
- Trend regime (ADX > 25): SMA crossover with ATR distance filter
- Range regime (ADX <= 25): Bollinger Bands mean-reversion
- Regime classification: ADX

Dashboard features:
- Real-time account monitoring (refreshes every 30s)
- Interval-based strategy analysis (default: cym's 3-hour live session)
- Performance metrics: Sharpe, Sortino, Calmar, MDD, VaR(95%), Win Rate, etc.
- Strategy PnL vs Benchmark (EUR/USD buy-and-hold)
- Kill Switch with double-confirmation
"""

import os
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timezone, timedelta

# =============================================================================
# 0. GLOBAL STRATEGY STATE (checked by bot each cycle)
# =============================================================================
strategy_running = True   # Set to False by Pause button; bot skips orders when False

# =============================================================================
# 1. CONFIG
# =============================================================================
API_TOKEN  = "7ace9b7dadae10843dfabe1fc498e91c-0edceaab79015427ab0fa8f820f3bc0b"
ACCOUNT_ID = "101-003-38996526-001"
INSTRUMENT = "EUR_USD"
GRANULARITY = "M5"
REST_URL   = "https://api-fxpractice.oanda.com"
HEADERS    = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type":  "application/json"
}

# Strategy parameters (same as cym's notebook)
FAST_SMA       = 10
SLOW_SMA       = 30
ATR_PERIOD     = 14
TRANSACTION_COST = 0.0001
TRADE_UNITS    = 1000

# Default interval = cym's 3-hour live session (UTC)
DEFAULT_START = "2026-04-06 15:49"
DEFAULT_END   = "2026-04-06 18:48"
WARMUP_BARS   = 50          # extra bars prepended for indicator warm-up
REFRESH_MS    = 30_000      # real-time panel refresh interval (ms)

# =============================================================================
# 2. OANDA API HELPERS
# =============================================================================
def oanda_get(endpoint, params=None):
    url = f"{REST_URL}{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if not r.ok:
        raise RuntimeError(f"GET {url} → {r.status_code}: {r.text[:200]}")
    return r.json()

def oanda_post(endpoint, payload):
    url = f"{REST_URL}{endpoint}"
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    if not r.ok:
        raise RuntimeError(f"POST {url} → {r.status_code}: {r.text[:200]}")
    return r.json()

def get_account_summary():
    return oanda_get(f"/v3/accounts/{ACCOUNT_ID}/summary")["account"]

def get_open_positions():
    return oanda_get(f"/v3/accounts/{ACCOUNT_ID}/openPositions").get("positions", [])

def get_open_trades():
    return oanda_get(f"/v3/accounts/{ACCOUNT_ID}/openTrades").get("trades", [])

def close_all_positions():
    """Close all open positions for INSTRUMENT."""
    results = []
    for pos in get_open_positions():
        if pos.get("instrument") != INSTRUMENT:
            continue
        long_units  = float(pos.get("long",  {}).get("units", "0"))
        short_units = float(pos.get("short", {}).get("units", "0"))
        ep = f"/v3/accounts/{ACCOUNT_ID}/positions/{INSTRUMENT}/close"
        if long_units > 0:
            results.append(oanda_post(ep, {"longUnits":  "ALL"}))
        if short_units < 0:
            results.append(oanda_post(ep, {"shortUnits": "ALL"}))
    return results

# =============================================================================
# 3. CANDLE FETCHING (with time-range support)
# =============================================================================
def fetch_candles_range(start_dt: datetime, end_dt: datetime,
                        granularity=GRANULARITY, instrument=INSTRUMENT) -> pd.DataFrame:
    """
    Fetch M5 candles from OANDA for [start_dt, end_dt] (UTC-aware datetimes).
    Prepends WARMUP_BARS extra candles before start_dt for indicator warm-up,
    then returns the full dataframe (caller should slice if needed).
    """
    warmup_start = start_dt - timedelta(minutes=5 * WARMUP_BARS)

    params = {
        "granularity": granularity,
        "from":  warmup_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to":    end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "price": "M",
    }
    raw = oanda_get(f"/v3/instruments/{instrument}/candles", params=params)

    rows = []
    for c in raw.get("candles", []):
        if not c.get("complete", False):
            continue
        mid = c.get("mid", {})
        rows.append({
            "Time":   c["time"],
            "Open":   float(mid["o"]),
            "High":   float(mid["h"]),
            "Low":    float(mid["l"]),
            "Close":  float(mid["c"]),
            "Volume": c.get("volume", np.nan),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df = df.set_index("Time").sort_index()
    return df

# =============================================================================
# 4. STRATEGY LOGIC (identical to cym's notebook)
# =============================================================================
def compute_indicators(df):
    df = df.copy()
    df["SMA_F"] = df["Close"].rolling(FAST_SMA).mean()
    df["SMA_S"] = df["Close"].rolling(SLOW_SMA).mean()

    hl   = df["High"] - df["Low"]
    hcp  = np.abs(df["High"] - df["Close"].shift(1))
    lcp  = np.abs(df["Low"]  - df["Close"].shift(1))
    tr   = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()

    plus_dm  = df["High"].diff().clip(lower=0)
    minus_dm = df["Low"].diff().clip(upper=0).abs()
    atr_s    = tr.rolling(ATR_PERIOD).mean()
    plus_di  = 100 * (plus_dm.rolling(ATR_PERIOD).mean()  / atr_s)
    minus_di = 100 * (minus_dm.rolling(ATR_PERIOD).mean() / atr_s)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["ADX"] = dx.rolling(ATR_PERIOD).mean()

    df["Std"]   = df["Close"].rolling(SLOW_SMA).std()
    df["Upper"] = df["SMA_S"] + 2 * df["Std"]
    df["Lower"] = df["SMA_S"] - 2 * df["Std"]
    return df

def generate_signal(row):
    """Return (signal, reason) for a single indicator row."""
    if pd.isna(row["ADX"]) or pd.isna(row["ATR"]) or \
       pd.isna(row["SMA_F"]) or pd.isna(row["SMA_S"]):
        return 0, "NOT_READY"

    filt = 0.2 * row["ATR"]

    if row["ADX"] > 25:
        if row["SMA_F"] > row["SMA_S"] and row["Close"] > row["SMA_S"] + filt:
            return  1, "TREND_LONG"
        elif row["SMA_F"] < row["SMA_S"] and row["Close"] < row["SMA_S"] - filt:
            return -1, "TREND_SHORT"
        else:
            return  0, "TREND_NO_TRADE"
    else:
        if row["Close"] > row["Upper"]:
            return -1, "RANGE_SHORT"
        elif row["Close"] < row["Lower"]:
            return  1, "RANGE_LONG"
        else:
            return  0, "RANGE_NO_TRADE"

def run_backtest(df_full: pd.DataFrame, start_dt: datetime, end_dt: datetime):
    """
    Vectorised backtest over the target interval.
    Returns (df_target, trades_df, equity_series, metrics_dict).
    """
    df = compute_indicators(df_full)

    # Slice to target window only (warm-up already done)
    df_target = df.loc[
        (df.index >= start_dt) & (df.index <= end_dt)
    ].copy()

    if df_target.empty:
        return df_target, pd.DataFrame(), pd.Series(dtype=float), {}

    # Generate signals for every bar in the target window
    signals, reasons = [], []
    for _, row in df_target.iterrows():
        s, r = generate_signal(row)
        signals.append(s)
        reasons.append(r)
    df_target["Signal"]  = signals
    df_target["Reason"]  = reasons

    # Simulate PnL bar-by-bar
    position   = 0
    entry_price = None
    realized_pnl = 0.0
    equity_vals  = []
    trades = []   # list of (entry_time, exit_time, direction, pnl)

    for i, (t, row) in enumerate(df_target.iterrows()):
        sig = row["Signal"]
        price = row["Close"]

        # Determine action
        if sig == position:
            action = "HOLD"
        elif sig != 0 and position == 0:
            action = "OPEN"
        elif sig == 0 and position != 0:
            action = "CLOSE"
        else:
            action = "REVERSE"

        # Execute
        if action in ("CLOSE", "REVERSE") and position != 0:
            raw_ret = (price - entry_price) / entry_price
            if position == -1:
                raw_ret = -raw_ret
            pnl = raw_ret - TRANSACTION_COST
            realized_pnl += pnl
            trades.append({
                "exit_time": t,
                "direction": position,
                "pnl":       pnl,
            })
            position    = 0
            entry_price = None

        if action in ("OPEN", "REVERSE") and sig != 0:
            position    = sig
            entry_price = price

        # Unrealized PnL for equity curve
        if position != 0 and entry_price is not None:
            unreal = (price - entry_price) / entry_price
            if position == -1:
                unreal = -unreal
            unreal -= TRANSACTION_COST
        else:
            unreal = 0.0

        equity_vals.append(realized_pnl + unreal)

    equity = pd.Series(equity_vals, index=df_target.index, name="Strategy")
    trades_df = pd.DataFrame(trades)

    # Benchmark: EUR/USD buy-and-hold over the interval
    p0 = df_target["Close"].iloc[0]
    benchmark = (df_target["Close"] - p0) / p0

    metrics = compute_metrics(equity, trades_df, benchmark)

    df_target["Equity"]    = equity.values
    df_target["Benchmark"] = benchmark.values
    return df_target, trades_df, equity, metrics

# =============================================================================
# 5. PERFORMANCE METRICS
# =============================================================================
def compute_metrics(equity: pd.Series, trades_df: pd.DataFrame,
                    benchmark: pd.Series) -> dict:
    returns = equity.diff().dropna()

    # Annualisation factor: M5 bars per year
    bars_per_year = 252 * 24 * 12

    def safe_div(a, b):
        return float(a / b) if b and not np.isnan(b) else np.nan

    # --- Total Return ---
    total_return = float(equity.iloc[-1]) if len(equity) else np.nan

    # --- Sharpe ---
    sharpe = safe_div(
        returns.mean() * np.sqrt(bars_per_year),
        returns.std()
    )

    # --- Sortino ---
    down = returns[returns < 0]
    sortino = safe_div(
        returns.mean() * np.sqrt(bars_per_year),
        down.std()
    )

    # --- Max Drawdown ---
    wealth = 1 + equity          # convert cumulative return to wealth curve
    cum = wealth.cummax()
    dd  = (wealth - cum) / cum
    max_dd = float(dd.min())

    # --- Calmar ---
    annualised_return = returns.mean() * bars_per_year
    calmar = safe_div(annualised_return, abs(max_dd)) if abs(max_dd) > 0.01 else np.nan

    # --- VaR 95% ---
    var_95 = float(np.percentile(returns, 5)) if len(returns) >= 20 else np.nan

    # --- Win Rate & Trade Count ---
    if not trades_df.empty and "pnl" in trades_df.columns:
        total_trades = len(trades_df)
        win_rate     = float((trades_df["pnl"] > 0).sum() / total_trades)
    else:
        total_trades = 0
        win_rate     = np.nan

    # --- Benchmark Return ---
    bm_return = float(benchmark.iloc[-1]) if len(benchmark) else np.nan

    return {
        "Total Return":      f"{total_return:.4%}" if not np.isnan(total_return) else "N/A",
        "Benchmark Return":  f"{bm_return:.4%}"    if not np.isnan(bm_return)    else "N/A",
        "Sharpe Ratio":      f"{sharpe:.3f}"        if not np.isnan(sharpe)       else "N/A",
        "Sortino Ratio":     f"{sortino:.3f}"       if not np.isnan(sortino)      else "N/A",
        "Calmar Ratio":      f"{calmar:.3f}"        if not np.isnan(calmar)       else "N/A",
        "Max Drawdown":      f"{max_dd:.4%}"        if not np.isnan(max_dd)       else "N/A",
        "VaR (95%)":         f"{var_95:.6f}"        if not np.isnan(var_95)       else "N/A",
        "Win Rate":          f"{win_rate:.2%}"      if not np.isnan(win_rate)     else "N/A",
        "Total Trades":      str(total_trades),
    }

# =============================================================================
# 6. DASH APP LAYOUT
# =============================================================================
app = dash.Dash(__name__, title="FT5010 Group 6 — Strategy Dashboard")

# --- colour palette ---
C = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "accent":    "#58a6ff",
    "green":     "#3fb950",
    "red":       "#f85149",
    "yellow":    "#d29922",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
    "card":      "#1c2128",
}

def card(title, value_id, color=C["text"], size="22px"):
    return html.Div([
        html.P(title, style={"margin": 0, "fontSize": "11px",
                             "color": C["muted"], "letterSpacing": "0.05em",
                             "textTransform": "uppercase"}),
        html.P(id=value_id, children="—",
               style={"margin": "2px 0 0 0", "fontSize": size,
                      "fontWeight": "700", "color": color,
                      "fontFamily": "monospace"}),
    ], style={"background": C["card"], "border": f"1px solid {C['border']}",
              "borderRadius": "6px", "padding": "8px 12px"})

app.layout = html.Div(style={
    "background": C["bg"], "fontFamily": "'Segoe UI', sans-serif",
    "color": C["text"],
    "width": "1920px", "height": "1080px", "overflow": "hidden",
    "margin": "0 auto", "padding": "14px 20px",
    "boxSizing": "border-box",
}, children=[

    # ── Auto-refresh interval ──────────────────────────────────────────────
    dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0),

    # ── Stores ────────────────────────────────────────────────────────────
    dcc.Store(id="kill-confirm-store", data={"confirmed": False}),
    dcc.Store(id="trading-store", data={"active": True}),
    dcc.Store(id="sim-store", data={"active": True, "direction": 0,
                                     "entry_price": None, "entry_time": None}),

    # ══════════════════════════════════════════════════════════════════════
    # HEADER (40px)
    # ══════════════════════════════════════════════════════════════════════
    html.Div(style={"borderBottom": f"1px solid {C['border']}",
                    "paddingBottom": "6px", "marginBottom": "8px",
                    "display": "flex", "justifyContent": "space-between",
                    "alignItems": "flex-end", "height": "40px"}, children=[
        html.Div([
            html.H1("FT5010 Group 6 — Live Strategy Dashboard",
                    style={"margin": 0, "fontSize": "18px",
                           "color": C["accent"], "fontWeight": "700"}),
            html.P("EUR/USD · M5 · Hybrid Trend (SMA/ATR/ADX) + Mean-Reversion (Bollinger Bands)",
                   style={"margin": "2px 0 0 0", "fontSize": "11px",
                          "color": C["muted"]}),
        ]),
        html.P(id="header-time", style={"fontSize": "11px",
                                        "color": C["muted"],
                                        "fontFamily": "monospace"}),
    ]),

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — STRATEGY BACKTEST ANALYSIS (~620px)
    # ══════════════════════════════════════════════════════════════════════
    html.H2("Strategy Backtest Analysis",
            style={"fontSize": "13px", "color": C["muted"],
                   "letterSpacing": "0.08em", "textTransform": "uppercase",
                   "margin": "0 0 6px 0", "fontWeight": "600"}),

    # Time picker row
    html.Div(style={"display": "flex", "alignItems": "center",
                    "gap": "10px", "marginBottom": "6px"}, children=[
        html.Span("Start (UTC):", style={"color": C["muted"], "fontSize": "12px"}),
        dcc.Input(id="start-input", type="text", value=DEFAULT_START,
                  placeholder="YYYY-MM-DD HH:MM",
                  style={"background": C["card"], "color": C["text"],
                         "border": f"1px solid {C['border']}",
                         "borderRadius": "4px", "padding": "4px 8px",
                         "fontSize": "12px", "fontFamily": "monospace",
                         "width": "160px"}),
        html.Span("End (UTC):", style={"color": C["muted"], "fontSize": "12px"}),
        dcc.Input(id="end-input", type="text", value=DEFAULT_END,
                  placeholder="YYYY-MM-DD HH:MM",
                  style={"background": C["card"], "color": C["text"],
                         "border": f"1px solid {C['border']}",
                         "borderRadius": "4px", "padding": "4px 8px",
                         "fontSize": "12px", "fontFamily": "monospace",
                         "width": "160px"}),
        html.Button("▶  Run Analysis", id="run-btn", n_clicks=0,
                    style={"background": C["accent"], "color": "#0d1117",
                           "border": "none", "borderRadius": "4px",
                           "padding": "5px 16px", "fontWeight": "700",
                           "fontSize": "12px", "cursor": "pointer"}),
        html.Span(id="run-status", style={"color": C["muted"], "fontSize": "11px"}),
    ]),

    # Metrics row
    html.Div(id="metrics-row",
             style={"display": "grid",
                    "gridTemplateColumns": "repeat(9, 1fr)",
                    "gap": "6px", "marginBottom": "6px"}),

    # Charts
    dcc.Graph(id="price-chart",
              config={"displayModeBar": False},
              style={"background": C["panel"],
                     "borderRadius": "6px", "marginBottom": "6px"}),

    dcc.Graph(id="pnl-chart",
              config={"displayModeBar": False},
              style={"background": C["panel"],
                     "borderRadius": "6px", "marginBottom": "10px"}),

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — LIVE MONITOR (~380px)
    # ══════════════════════════════════════════════════════════════════════
    html.Div(style={"borderTop": f"1px solid {C['border']}",
                    "paddingTop": "8px"}, children=[

        html.H2("Live Monitor",
                style={"fontSize": "13px", "color": C["muted"],
                       "letterSpacing": "0.08em", "textTransform": "uppercase",
                       "margin": "0 0 6px 0", "fontWeight": "600"}),

        # Row 1: account cards + position cards (7 columns)
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "repeat(7, 1fr)",
                        "gap": "8px", "marginBottom": "8px"}, children=[
            card("Balance (SGD)",       "rt-balance"),
            card("NAV (Equity)",        "rt-nav",       C["accent"]),
            card("Margin Used",         "rt-margin-used"),
            card("Margin Available",    "rt-margin-avail", C["green"]),
            card("Position",            "rt-position"),
            card("Signal Reason",       "rt-signal-reason"),
            card("Unrealized PnL",      "rt-unrealized"),
        ]),

        # Row 2: buttons + mini chart side by side
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "320px 1fr",
                        "gap": "10px"}, children=[

            # Left: buttons stacked
            html.Div(style={"display": "flex", "flexDirection": "column",
                            "gap": "6px"}, children=[

                # Kill Switch
                html.Div([
                    html.Button(id="kill-btn", children="⬛  CLOSE ALL POSITIONS",
                                n_clicks=0,
                                style={"width": "100%",
                                       "background": C["red"], "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "padding": "8px", "fontSize": "12px",
                                       "fontWeight": "700", "cursor": "pointer"}),
                    html.P(id="kill-status", children="",
                           style={"margin": "3px 0 0", "fontSize": "10px",
                                  "color": C["yellow"], "minHeight": "14px"}),
                ], style={"background": C["card"],
                          "border": f"1px solid {C['red']}30",
                          "borderRadius": "6px", "padding": "8px"}),

                # Start/Stop Trading
                html.Div([
                    html.Button(id="trading-btn", children="⬛  STOP TRADING",
                                n_clicks=0,
                                style={"width": "100%",
                                       "background": C["red"], "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "padding": "8px", "fontSize": "12px",
                                       "fontWeight": "700", "cursor": "pointer"}),
                    html.P(id="trading-status",
                           children="Trading is ACTIVE — auto-opened on load",
                           style={"margin": "3px 0 0", "fontSize": "10px",
                                  "color": C["green"], "minHeight": "14px"}),
                ], style={"background": C["card"],
                          "border": f"1px solid {C['green']}30",
                          "borderRadius": "6px", "padding": "8px"}),

                # Sim info
                html.P(id="sim-status",
                       children="Initialising simulated position...",
                       style={"fontSize": "10px", "color": C["muted"],
                              "margin": "0", "padding": "4px 8px"}),
            ]),

            # Right: mini chart
            dcc.Graph(id="live-mini-chart",
                      config={"displayModeBar": False},
                      style={"background": C["panel"], "borderRadius": "6px"}),
        ]),
    ]),

])  # end layout

# =============================================================================
# 7. CALLBACKS — REAL-TIME PANEL (auto-opens sim on first tick)
# =============================================================================
@app.callback(
    Output("header-time",       "children"),
    Output("rt-balance",        "children"),
    Output("rt-nav",            "children"),
    Output("rt-margin-used",    "children"),
    Output("rt-margin-avail",   "children"),
    Output("rt-position",       "children"),
    Output("rt-signal-reason",  "children"),
    Output("rt-unrealized",     "children"),
    Output("live-mini-chart",   "figure"),
    Output("sim-store",         "data"),
    Output("sim-status",        "children"),
    Input("refresh-interval",   "n_intervals"),
    State("sim-store",          "data"),
    State("trading-store",      "data"),
)
def update_realtime(_, sim, trading):
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        acc = get_account_summary()
        balance       = f"{float(acc.get('balance',      0)):.2f}"
        nav           = f"{float(acc.get('NAV',          0)):.2f}"
        margin_used   = f"{float(acc.get('marginUsed',   0)):.2f}"
        margin_avail  = f"{float(acc.get('marginAvailable', 0)):.2f}"
    except Exception:
        balance = nav = margin_used = margin_avail = "ERR"

    # Fetch recent candles (used for signal + mini chart)
    latest_price = None
    df_recent = pd.DataFrame()
    sig, reason = 0, "N/A"
    try:
        df_recent = fetch_candles_range(
            datetime.now(timezone.utc) - timedelta(hours=2),
            datetime.now(timezone.utc),
        )
        if not df_recent.empty:
            latest_price = float(df_recent["Close"].iloc[-1])
        df_ind = compute_indicators(df_recent)
        sig, reason = generate_signal(df_ind.iloc[-1])
    except Exception:
        pass

    # Auto-open sim position on first tick (or if trading active but no entry)
    is_trading = trading and trading.get("active", True)
    if is_trading and sim and sim.get("active") and sim.get("entry_price") is None:
        if latest_price is not None and sig != 0:
            sim = {
                "active": True,
                "direction": sig,
                "entry_price": latest_price,
                "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            }

    # Position & Unrealized PnL
    if sim and sim.get("active") and sim.get("entry_price") is not None:
        direction = sim["direction"]
        entry_px  = sim["entry_price"]
        dir_label = "SHORT" if direction == -1 else "LONG"
        position_str = f"{dir_label} (SIM)"
        if latest_price is not None:
            raw_pnl = (latest_price - entry_px) * direction
            unrealized = f"{raw_pnl:.6f}"
        else:
            unrealized = "N/A"
        sim_text = f"SIM {dir_label} @ {entry_px:.5f} ({sim.get('entry_time', '')} UTC)"
    else:
        position_str = "FLAT"
        unrealized = "0.000000"
        sim_text = "No simulated position" if is_trading else "Trading stopped"

    # ── Mini chart: last 20 M5 candles ────────────────────────────────────
    mini_fig = go.Figure()
    if not df_recent.empty:
        df_tail = df_recent.tail(20)
        mini_fig.add_trace(go.Scatter(
            x=df_tail.index, y=df_tail["Close"],
            name="EUR/USD", line=dict(color=C["accent"], width=2),
        ))
        if sim and sim.get("active") and sim.get("entry_price") is not None:
            mini_fig.add_hline(
                y=sim["entry_price"],
                line=dict(color=C["red"] if sim["direction"] == -1 else C["green"],
                          width=2, dash="dash"),
                annotation_text=f"Entry {sim['entry_price']:.5f}",
                annotation_font=dict(color=C["text"], size=10),
            )
    mini_fig.update_layout(
        title=dict(text="EUR/USD Live — Last 20 M5 Bars",
                   font=dict(color=C["text"], size=12)),
        paper_bgcolor=C["panel"], plot_bgcolor=C["bg"],
        font=dict(color=C["muted"], size=10),
        xaxis=dict(showgrid=False, color=C["muted"]),
        yaxis=dict(showgrid=True, gridcolor=C["border"], color=C["muted"],
                   tickformat=".5f"),
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )

    return (now_str, balance, nav, margin_used, margin_avail,
            position_str, reason, unrealized, mini_fig, sim, sim_text)

# =============================================================================
# 8. CALLBACKS — KILL SWITCH (double-confirmation)
# =============================================================================
@app.callback(
    Output("kill-btn",          "children"),
    Output("kill-btn",          "style"),
    Output("kill-status",       "children"),
    Output("kill-confirm-store","data"),
    Input("kill-btn",           "n_clicks"),
    State("kill-confirm-store", "data"),
    prevent_initial_call=True,
)
def kill_switch(n_clicks, store):
    base_style = {
        "marginTop": "8px", "width": "100%",
        "border": "none", "borderRadius": "6px",
        "padding": "12px", "fontSize": "0.85rem",
        "fontWeight": "700", "cursor": "pointer",
        "letterSpacing": "0.04em", "color": "white",
    }

    if not store.get("confirmed"):
        # First click → ask for confirmation
        return (
            "⚠️  Confirm? Click Again to Close All",
            {**base_style, "background": C["yellow"]},
            "Click once more to execute.",
            {"confirmed": True},
        )
    else:
        # Second click → execute
        try:
            close_all_positions()
            msg = "✅ All positions closed."
        except Exception as e:
            msg = f"❌ Error: {str(e)[:60]}"
        return (
            "⬛  CLOSE ALL POSITIONS",
            {**base_style, "background": C["red"]},
            msg,
            {"confirmed": False},
        )

# =============================================================================
# 8b. CALLBACKS — START / STOP TRADING
# =============================================================================
@app.callback(
    Output("trading-btn",    "children"),
    Output("trading-btn",    "style"),
    Output("trading-status", "children"),
    Output("trading-store",  "data"),
    Input("trading-btn",     "n_clicks"),
    State("trading-store",   "data"),
    prevent_initial_call=True,
)
def toggle_trading(n_clicks, store):
    global strategy_running

    active = store.get("active", True)
    base_style = {
        "width": "100%", "border": "none", "borderRadius": "4px",
        "padding": "8px", "fontSize": "12px",
        "fontWeight": "700", "cursor": "pointer",
    }

    if active:
        # Currently trading → stop
        strategy_running = False
        return (
            "▶  START TRADING",
            {**base_style, "background": C["green"], "color": "#0d1117"},
            "Trading STOPPED — signals recorded but no orders",
            {"active": False},
        )
    else:
        # Currently stopped → start
        strategy_running = True
        return (
            "⬛  STOP TRADING",
            {**base_style, "background": C["red"], "color": "white"},
            "Trading is ACTIVE",
            {"active": True},
        )

# =============================================================================
# 9. CALLBACKS — INTERVAL ANALYSIS
# =============================================================================
@app.callback(
    Output("metrics-row",  "children"),
    Output("price-chart",  "figure"),
    Output("pnl-chart",    "figure"),
    Output("run-status",   "children"),
    Input("run-btn",       "n_clicks"),
    State("start-input",   "value"),
    State("end-input",     "value"),
    prevent_initial_call=False,
)
def run_analysis(n_clicks, start_str, end_str):
    # Parse times
    try:
        start_dt = datetime.strptime(start_str.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end_str.strip(),   "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except Exception:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor=C["panel"], plot_bgcolor=C["panel"])
        return [], empty_fig, empty_fig, "⚠️ Invalid date format. Use YYYY-MM-DD HH:MM"

    # Fetch data
    try:
        df_full = fetch_candles_range(start_dt, end_dt)
        if df_full.empty:
            raise ValueError("No candles returned.")
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor=C["panel"], plot_bgcolor=C["panel"])
        return [], empty_fig, empty_fig, f"❌ API error: {str(e)[:80]}"

    # Run backtest
    df_target, trades_df, equity, metrics = run_backtest(df_full, start_dt, end_dt)

    if df_target.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor=C["panel"], plot_bgcolor=C["panel"])
        return [], empty_fig, empty_fig, "⚠️ No data in selected interval."

    # ── Metrics cards ──────────────────────────────────────────────────────
    def metric_card(label, value):
        color = C["green"] if (isinstance(value, str) and value.startswith("+")) else C["text"]
        return html.Div([
            html.P(label, style={"margin": 0, "fontSize": "10px",
                                  "color": C["muted"],
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.04em"}),
            html.P(value, style={"margin": "2px 0 0", "fontSize": "14px",
                                  "fontWeight": "700", "color": color,
                                  "fontFamily": "monospace"}),
        ], style={"background": C["card"],
                  "border": f"1px solid {C['border']}",
                  "borderRadius": "6px", "padding": "6px 10px"})

    metrics_cards = [metric_card(k, v) for k, v in metrics.items()]

    # ── Price chart with signal annotations ───────────────────────────────
    long_mask  = df_target["Signal"] ==  1
    short_mask = df_target["Signal"] == -1

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=df_target.index, y=df_target["Close"],
        name="EUR/USD Close", line=dict(color=C["accent"], width=1.5)
    ))
    price_fig.add_trace(go.Scatter(
        x=df_target.index, y=df_target["SMA_F"],
        name=f"SMA {FAST_SMA}", line=dict(color=C["yellow"], width=1, dash="dot")
    ))
    price_fig.add_trace(go.Scatter(
        x=df_target.index, y=df_target["SMA_S"],
        name=f"SMA {SLOW_SMA}", line=dict(color=C["muted"], width=1, dash="dot")
    ))
    # Bollinger Bands (filled)
    price_fig.add_trace(go.Scatter(
        x=pd.concat([df_target.index.to_series(),
                     df_target.index.to_series()[::-1]]),
        y=pd.concat([df_target["Upper"], df_target["Lower"][::-1]]),
        fill="toself", fillcolor="rgba(88,166,255,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="Bollinger Bands",
        showlegend=True,
    ))
    # Long signals
    price_fig.add_trace(go.Scatter(
        x=df_target.index[long_mask],
        y=df_target["Close"][long_mask],
        mode="markers",
        marker=dict(symbol="triangle-up", size=10,
                    color=C["green"], line=dict(width=0)),
        name="Long Signal"
    ))
    # Short signals
    price_fig.add_trace(go.Scatter(
        x=df_target.index[short_mask],
        y=df_target["Close"][short_mask],
        mode="markers",
        marker=dict(symbol="triangle-down", size=10,
                    color=C["red"], line=dict(width=0)),
        name="Short Signal"
    ))

    # 实际交易动作（signal变化的那一刻）
    prev_sig = df_target["Signal"].shift(1)
    entry_long_mask  = (df_target["Signal"] ==  1) & (prev_sig !=  1)
    entry_short_mask = (df_target["Signal"] == -1) & (prev_sig != -1)
    exit_long_mask   = (df_target["Signal"] !=  1) & (prev_sig ==  1)  # 平多
    exit_short_mask  = (df_target["Signal"] != -1) & (prev_sig == -1)  # 平空

    price_fig.add_trace(go.Scatter(
        x=df_target.index[entry_long_mask],
        y=df_target["Close"][entry_long_mask],
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=16, color=C["green"],
                    line=dict(color="white", width=1)),
        text="OL", textposition="top center",
        textfont=dict(color=C["green"], size=11, family="monospace"),
        name="Open Long"
    ))

    price_fig.add_trace(go.Scatter(
        x=df_target.index[exit_long_mask],
        y=df_target["Close"][exit_long_mask],
        mode="markers+text",
        marker=dict(symbol="triangle-down", size=16, color=C["yellow"],
                    line=dict(color="white", width=1)),
        text="CL", textposition="bottom center",
        textfont=dict(color=C["yellow"], size=11, family="monospace"),
        name="Close Long"
    ))

    price_fig.add_trace(go.Scatter(
        x=df_target.index[entry_short_mask],
        y=df_target["Close"][entry_short_mask],
        mode="markers+text",
        marker=dict(symbol="triangle-down", size=16, color=C["red"],
                    line=dict(color="white", width=1)),
        text="OS", textposition="bottom center",
        textfont=dict(color=C["red"], size=11, family="monospace"),
        name="Open Short"
    ))

    price_fig.add_trace(go.Scatter(
        x=df_target.index[exit_short_mask],
        y=df_target["Close"][exit_short_mask],
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=16, color=C["yellow"],
                    line=dict(color="white", width=1)),
        text="CS", textposition="top center",
        textfont=dict(color=C["yellow"], size=11, family="monospace"),
        name="Close Short"
    ))

    price_fig.update_layout(
        title=dict(text="EUR/USD Price with Strategy Signals",
                   font=dict(color=C["text"], size=14)),
        paper_bgcolor=C["panel"], plot_bgcolor=C["bg"],
        font=dict(color=C["muted"]),
        xaxis=dict(showgrid=False, color=C["muted"],
                   rangebreaks=[dict(bounds=["sat", "mon"])]),
        yaxis=dict(showgrid=True, gridcolor=C["border"], color=C["muted"]),
        legend=dict(bgcolor=C["panel"], bordercolor=C["border"],
                    borderwidth=1, font=dict(size=11)),
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )

    # ── PnL vs Benchmark chart ─────────────────────────────────────────────
    pnl_fig = go.Figure()
    pnl_fig.add_trace(go.Scatter(
        x=df_target.index, y=df_target["Equity"],
        name="Strategy PnL", fill="tozeroy",
        fillcolor="rgba(63,185,80,0.12)",
        line=dict(color=C["green"], width=2)
    ))
    pnl_fig.add_trace(go.Scatter(
        x=df_target.index, y=df_target["Benchmark"],
        name="Benchmark (EUR/USD Buy & Hold)",
        line=dict(color=C["yellow"], width=1.5, dash="dash")
    ))
    pnl_fig.add_hline(y=0, line=dict(color=C["border"], width=1))
    pnl_fig.update_layout(
        title=dict(text="Cumulative PnL — Strategy vs Benchmark",
                   font=dict(color=C["text"], size=14)),
        paper_bgcolor=C["panel"], plot_bgcolor=C["bg"],
        font=dict(color=C["muted"]),
        xaxis=dict(showgrid=False, color=C["muted"],
                   rangebreaks=[dict(bounds=["sat", "mon"])]),
        yaxis=dict(showgrid=True, gridcolor=C["border"],
                   color=C["muted"], tickformat=".5f"),
        legend=dict(bgcolor=C["panel"], bordercolor=C["border"],
                    borderwidth=1, font=dict(size=11)),
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )

    status = f"✅ Loaded {len(df_target)} bars · {len(trades_df)} trades"
    return metrics_cards, price_fig, pnl_fig, status

# =============================================================================
# 10. ENTRY POINT
# =============================================================================
server = app.server   # expose for gunicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
