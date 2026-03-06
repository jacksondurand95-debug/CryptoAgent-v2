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
TRADING_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD",
    "PEPE-USD", "SHIB-USD", "SUI-USD", "NEAR-USD", "RENDER-USD", "FET-USD",
    "INJ-USD", "TIA-USD", "SEI-USD", "WIF-USD",
]

# Multi-timeframe analysis
PRIMARY_TIMEFRAME = "SIX_HOUR"
TREND_TIMEFRAME = "ONE_DAY"
FAST_TIMEFRAME = "ONE_HOUR"
SCALP_TIMEFRAME = "FIFTEEN_MINUTE"

# Risk params — FULL SEND MODE ($3000 deployed)
MAX_POSITION_PCT = 0.40        # 40% per trade (~$1200) — SIZE UP
MAX_OPEN_POSITIONS = 5         # 5 concurrent (up to $6000 notional)
STOP_LOSS_ATR_MULT = 1.5      # Tight stops — cut losers FAST
TRAILING_STOP_ATR_MULT = 1.8  # 1.8x ATR trailing — lock gains sooner
TRAILING_ACTIVATION_R = 0.8   # Activate trailing earlier
TAKE_PROFIT_ATR_MULT = 6.0    # 6x ATR target — let winners RIP
MIN_CONFIDENCE = 0.50          # Lower bar — more trades, more action
MIN_EXPECTED_MOVE_PCT = 2.5    # 2.5% min (was 3%) — more opportunities
TIME_STOP_HOURS = 72           # 3 days max hold — rotate capital faster
MIN_HOLD_MINUTES = 20          # 20 min minimum — faster exits
REENTRY_COOLDOWN_MINUTES = 10  # 10 min cooldown — rapid reentry

# Fees — Coinbase Advanced Trade (Coinbase One does NOT cover Advanced Trade)
# Intro 1 tier: 0.60% maker / 1.20% taker
# STRATEGY: ALL orders are post_only limit = 0.60% guaranteed. NEVER market orders.
MAKER_FEE_PCT = 0.006          # 0.60% — the ONLY fee we should ever pay
TAKER_FEE_PCT = 0.012          # 1.20% — AVOID: never use market orders
ROUND_TRIP_FEE_PCT = 0.012     # 1.20% round trip (limit both sides)
ROUND_TRIP_FEE_TAKER_PCT = 0.012   # Same — we don't use market exits anymore
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

# Underground alpha feeds — free public APIs for edge
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"
ARKHAM_BASE = "https://api.arkhamintelligence.com"
DEXSCREENER_BASE = "https://api.dexscreener.com/latest"
DEFILLAMA_BASE = "https://api.llama.fi"
ALTERNATIVE_ME_BASE = "https://api.alternative.me"
WHALE_ALERT_WS = "wss://ws.whale-alert.io"

# Multi-exchange data
BYBIT_API_BASE = "https://api.bybit.com"
OKX_API_BASE = "https://www.okx.com"
BINANCE_API_BASE = "https://fapi.binance.com"

# Strategy params
MOMENTUM_LOOKBACK = 28
MOMENTUM_THRESHOLD = 0.015
SHARPE_FILTER_LONG = 1.0
SQUEEZE_MIN_BARS = 3
