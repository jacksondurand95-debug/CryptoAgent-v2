"""CryptoAgent v3.0 — Beast Mode Trading Engine.

Dual-brain (Claude/Grok) with Coinbase One zero-fee trading.
Multi-exchange derivatives intel. Real-time market awareness.
Aggressive but disciplined — every 10 minutes, 24/7.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Coinbase
COINBASE_KEY_FILE = PROJECT_DIR / "coinbase_key.json"

# Trading pairs — expanded for more opportunities
TRADING_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]

# Multi-timeframe analysis
PRIMARY_TIMEFRAME = "SIX_HOUR"    # Main signal generation
TREND_TIMEFRAME = "ONE_DAY"       # Trend filter
FAST_TIMEFRAME = "ONE_HOUR"       # Entry timing + fast momentum
SCALP_TIMEFRAME = "FIFTEEN_MINUTE"  # Microstructure reads

# Risk params — AGGRESSIVE (Coinbase One = reduced fees)
MAX_POSITION_PCT = 0.30        # 30% per trade (bigger swings)
MAX_OPEN_POSITIONS = 3         # 3 concurrent — more exposure
STOP_LOSS_ATR_MULT = 2.0      # 2x ATR — tighter stops, faster rotation
TRAILING_STOP_ATR_MULT = 2.0  # Tighter trail for locking profits
TRAILING_ACTIVATION_R = 0.8   # Activate trail at 0.8R (protect gains earlier)
TAKE_PROFIT_ATR_MULT = 4.0    # 4x ATR — still hold for moves but take profits
MIN_CONFIDENCE = 0.55          # Lower threshold = more trades
MIN_EXPECTED_MOVE_PCT = 1.2    # Lower bar with reduced fees
TIME_STOP_HOURS = 72           # 3 days max — rotate capital faster
MIN_HOLD_MINUTES = 60          # 1 hour minimum hold
REENTRY_COOLDOWN_MINUTES = 30  # 30 min cooldown — faster reentry

# Fees — Coinbase One Preferred (25% rebate on Advanced Trade fees)
# Base tier: 0.60% maker / 1.20% taker
# After 25% rebate: ~0.45% maker / ~0.90% taker
# Using post_only limit orders for maker fee
MAKER_FEE_PCT = 0.0045         # 0.45% effective maker (after Coinbase One rebate)
TAKER_FEE_PCT = 0.009          # 0.90% effective taker (after rebate)
ROUND_TRIP_FEE_PCT = 0.009     # 0.90% round trip with limit orders (after rebate)
FEE_AUTO_DETECT = True         # Auto-detect from transaction_summary API

# Safety
MAX_DRAWDOWN_PCT = 0.40         # 40% drawdown warning
DAILY_MAX_LOSS_PCT = 0.15       # 15% daily loss limit — prevent blowup days
MAX_CONSECUTIVE_LOSSES = 5      # 5 losses in a row = pause 1 hour
COOLDOWN_AFTER_STOP_SEC = 30    # 30 sec cooldown after stop

# State (file-based, committed by Actions workflow)
STATE_FILE = PROJECT_DIR / "state.json"

# AI — Dual Brain (Claude primary, Grok fallback)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6-20250514")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4-0709")

# Legacy
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# TradingView
TV_ENABLED = True

# News & Intelligence
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")

# Analysis interval (GitHub Actions runs every 10 min)
ANALYSIS_INTERVAL_SEC = 600

# Data sources — multi-exchange derivatives
BYBIT_API_BASE = "https://api.bybit.com"
OKX_API_BASE = "https://www.okx.com"
BINANCE_API_BASE = "https://fapi.binance.com"

# AdaptiveTrend specific params
MOMENTUM_LOOKBACK = 28         # 28-period momentum lookback (6H candles = 7 days)
MOMENTUM_THRESHOLD = 0.015    # 1.5% momentum threshold (lower with reduced fees)
SHARPE_FILTER_LONG = 1.0      # Sharpe >= 1.0 for longs (less restrictive)
SQUEEZE_MIN_BARS = 3           # Min bars in squeeze before breakout
