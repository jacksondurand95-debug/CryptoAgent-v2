"""CryptoAgent v2 — Proven Algorithm Configuration.

Based on:
- AdaptiveTrend paper (arXiv 2602.11708): Sharpe 2.41, 6H timeframe
- Bollinger-Keltner Squeeze research: fewer but bigger trades
- Fee-aware optimization: must clear 0.80% round-trip
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Coinbase
COINBASE_KEY_FILE = PROJECT_DIR / "coinbase_key.json"

# Trading pairs
TRADING_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD"]

# PRIMARY TIMEFRAME: 6H (from AdaptiveTrend paper — filters noise, bigger moves)
PRIMARY_TIMEFRAME = "SIX_HOUR"
TREND_TIMEFRAME = "ONE_DAY"
FAST_TIMEFRAME = "ONE_HOUR"

# Risk params — PROVEN (from AdaptiveTrend paper ablation study)
MAX_POSITION_PCT = 0.25        # 25% per trade (2 concurrent max)
MAX_OPEN_POSITIONS = 2         # Focus capital, not scatter
STOP_LOSS_ATR_MULT = 2.5      # 2.5x ATR — paper's optimal (+0.73 Sharpe)
TRAILING_STOP_ATR_MULT = 2.5  # Dynamic trail at 2.5x ATR from peak
TRAILING_ACTIVATION_R = 1.0   # Activate trail at 1R profit (let it breathe)
TAKE_PROFIT_ATR_MULT = 5.0    # 5x ATR — hold for big moves
MIN_CONFIDENCE = 0.65          # High conviction only — fewer but better trades
MIN_EXPECTED_MOVE_PCT = 2.0    # Must be >2x round-trip fees (0.80%)
TIME_STOP_HOURS = 120          # 5 days before time-cutting
MIN_HOLD_MINUTES = 120         # 2 hour minimum hold
REENTRY_COOLDOWN_MINUTES = 60  # 1 hour cooldown between trades

# Fees (using LIMIT orders with post_only)
MAKER_FEE_PCT = 0.004          # 0.40% maker
TAKER_FEE_PCT = 0.006          # 0.60% taker (fallback)
ROUND_TRIP_FEE_PCT = 0.008     # 0.80% round trip with limit orders

# Safety — still aggressive but not suicidal
MAX_DRAWDOWN_PCT = 0.50         # 50% before warning (no halt)
DAILY_MAX_LOSS_PCT = 1.0        # Effectively disabled (100%)
MAX_CONSECUTIVE_LOSSES = 999    # Effectively disabled
COOLDOWN_AFTER_STOP_SEC = 60    # 1 min cooldown after stop

# State (file-based, committed by Actions workflow)
STATE_FILE = PROJECT_DIR / "state.json"

# AI
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# TradingView
TV_ENABLED = True

# News
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")

# Analysis interval (GitHub Actions runs every 10 min)
ANALYSIS_INTERVAL_SEC = 600

# AdaptiveTrend specific params
MOMENTUM_LOOKBACK = 28         # 28-period momentum lookback (6H candles = 7 days)
MOMENTUM_THRESHOLD = 0.02     # 2% momentum threshold for entry
SHARPE_FILTER_LONG = 1.3      # Prior-period Sharpe >= 1.3 for longs
SQUEEZE_MIN_BARS = 3           # Min bars in squeeze before breakout
