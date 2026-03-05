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

# Risk params — disciplined
MAX_POSITION_PCT = 0.25        # 25% per trade
MAX_OPEN_POSITIONS = 3         # 3 concurrent
STOP_LOSS_ATR_MULT = 2.0      # 2x ATR stops
TRAILING_STOP_ATR_MULT = 2.0  # 2x ATR trailing
TRAILING_ACTIVATION_R = 0.8   # Activate trail at 0.8R
TAKE_PROFIT_ATR_MULT = 4.0    # 4x ATR target
MIN_CONFIDENCE = 0.60          # Higher bar = fewer, better trades
MIN_EXPECTED_MOVE_PCT = 4.0    # Must clear 1.8% round-trip fees with margin
TIME_STOP_HOURS = 72           # 3 days max hold
MIN_HOLD_MINUTES = 60          # 1 hour minimum
REENTRY_COOLDOWN_MINUTES = 30  # 30 min cooldown

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

# Analysis interval
ANALYSIS_INTERVAL_SEC = 600

# Multi-exchange data
BYBIT_API_BASE = "https://api.bybit.com"
OKX_API_BASE = "https://www.okx.com"
BINANCE_API_BASE = "https://fapi.binance.com"

# Strategy params
MOMENTUM_LOOKBACK = 28
MOMENTUM_THRESHOLD = 0.015
SHARPE_FILTER_LONG = 1.0
SQUEEZE_MIN_BARS = 3
