"""CryptoAgent v3.1 — Rebuilt Trading Engine.

Quant strategies as pre-filter + LLM validation.
Correct fee structure. Risk limits enforced.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Coinbase
COINBASE_KEY_FILE = PROJECT_DIR / "coinbase_key.json"

# Trading pairs
TRADING_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]

# Multi-timeframe analysis
PRIMARY_TIMEFRAME = "SIX_HOUR"
TREND_TIMEFRAME = "ONE_DAY"
FAST_TIMEFRAME = "ONE_HOUR"
SCALP_TIMEFRAME = "FIFTEEN_MINUTE"

# Risk params — BULL RUSH MODE ($3000 deployed)
MAX_POSITION_PCT = 0.30        # 30% per trade (~$900)
MAX_OPEN_POSITIONS = 4         # 4 concurrent (up to $3600 deployed)
STOP_LOSS_ATR_MULT = 1.8      # Tighter stops — cut losers fast
TRAILING_STOP_ATR_MULT = 2.0  # 2x ATR trailing
TRAILING_ACTIVATION_R = 1.0   # Let winners RUN before trailing kicks in
TAKE_PROFIT_ATR_MULT = 5.0    # 5x ATR target — bigger swings to clear fees
MIN_CONFIDENCE = 0.55          # More aggressive entry threshold
MIN_EXPECTED_MOVE_PCT = 3.0    # Must clear 1.2-1.8% round-trip fees
TIME_STOP_HOURS = 96           # 4 days max hold — give trades room
MIN_HOLD_MINUTES = 30          # 30 min minimum (was 60)
REENTRY_COOLDOWN_MINUTES = 15  # 15 min cooldown (was 30)

# Fees — Coinbase Advanced Trade
# IMPORTANT: These are FALLBACK rates if auto-detection fails.
# Actual rates come from /api/v3/brokerage/transaction_summary at runtime.
# Coinbase One (if active) = 0% maker / 0% taker
# Intro 1 (detected 2026-03-05) = 0.60% maker / 1.20% taker
MAKER_FEE_PCT = 0.006          # 0.60% Intro 1 maker (fallback)
TAKER_FEE_PCT = 0.012          # 1.20% Intro 1 taker (fallback)
ROUND_TRIP_FEE_PCT = 0.012     # 1.20% round trip (maker both sides)
ROUND_TRIP_FEE_TAKER_PCT = 0.018   # 1.80% if exit is market order
FEE_AUTO_DETECT = True         # Override with detected rates from API

# Safety — ENFORCED
MAX_DRAWDOWN_PCT = 0.40         # 40% drawdown halt
DAILY_MAX_LOSS_PCT = 0.10       # 10% daily loss limit
MAX_CONSECUTIVE_LOSSES = 5      # 5 losses = 1 hour pause
COOLDOWN_AFTER_STOP_SEC = 30

# State
STATE_FILE = PROJECT_DIR / "state.json"

# AI — Dual Brain
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6-20250514")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4-0709")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# TradingView
TV_ENABLED = True

# News
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")

# Analysis interval — 5 min for faster signal capture
ANALYSIS_INTERVAL_SEC = 300

# Multi-exchange data
BYBIT_API_BASE = "https://api.bybit.com"
OKX_API_BASE = "https://www.okx.com"
BINANCE_API_BASE = "https://fapi.binance.com"

# Strategy params
MOMENTUM_LOOKBACK = 28
MOMENTUM_THRESHOLD = 0.015
SHARPE_FILTER_LONG = 1.0
SQUEEZE_MIN_BARS = 3
