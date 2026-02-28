"""CryptoAgent v2 — Ultra Aggressive Configuration."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Coinbase
COINBASE_KEY_FILE = PROJECT_DIR / "coinbase_key.json"

# Trading pairs
TRADING_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD"]

# AGGRESSIVE risk params
MAX_POSITION_PCT = 0.35        # 35% per trade
MAX_OPEN_POSITIONS = 5         # Up to 5 concurrent
STOP_LOSS_ATR_MULT = 1.5      # Tight stop: 1.5x ATR
TRAILING_STOP_ATR_MULT = 1.0  # Tight trail: 1x ATR
TRAILING_ACTIVATION_R = 0.5   # Activate trail early at 0.5R
TAKE_PROFIT_ATR_MULT = 3.0    # Take profit at 3x ATR
MIN_CONFIDENCE = 0.40          # Low bar — more trades
MIN_EXPECTED_MOVE_PCT = 1.0    # Low bar — more entries
TIME_STOP_HOURS = 24           # Cut losers faster
MIN_HOLD_MINUTES = 5           # Almost no hold restriction
REENTRY_COOLDOWN_MINUTES = 5   # Quick re-entry

# Fees (using LIMIT orders)
MAKER_FEE_PCT = 0.004          # 0.40% maker
TAKER_FEE_PCT = 0.006          # 0.60% taker (fallback)
ROUND_TRIP_FEE_PCT = 0.008     # 0.80% round trip with limit orders

# NO SAFETY BLOCKS
MAX_DRAWDOWN_PCT = 0.50         # 50% before warning (no halt)
DAILY_MAX_LOSS_PCT = 1.0        # Effectively disabled (100%)
MAX_CONSECUTIVE_LOSSES = 999    # Effectively disabled
COOLDOWN_AFTER_STOP_SEC = 10    # 10 seconds, basically nothing

# State
GIST_ID = os.environ.get("STATE_GIST_ID", "")
GITHUB_TOKEN = os.environ.get("GH_PAT", "")

# AI
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# TradingView
TV_ENABLED = True

# News
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")

# Analysis interval
ANALYSIS_INTERVAL_SEC = 60
