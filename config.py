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

# Risk params — OPTIMIZED FOR HIGH FEES (detected 0.4-0.8%)
MAX_POSITION_PCT = 0.20        # 20% per trade — conservative sizing given fees
MAX_OPEN_POSITIONS = 2         # 2 concurrent — focus on highest conviction only
STOP_LOSS_ATR_MULT = 2.5      # 2.5x ATR stops — wider to avoid fee churn
TRAILING_STOP_ATR_MULT = 2.0  # 2x ATR trailing — lock gains with room
TRAILING_ACTIVATION_R = 2.0   # Activate trailing after 2x risk — offset fees
TAKE_PROFIT_ATR_MULT = 10.0   # 10x ATR target — need bigger wins vs fees
MIN_CONFIDENCE = 0.65          # Higher bar (was 0.55) — quality over quantity
MIN_EXPECTED_MOVE_PCT = 4.0    # 4% min (was 3%) — must clear 5x round-trip fees
TIME_STOP_HOURS = 72           # 3 days max hold — reduce position churn
MIN_HOLD_MINUTES = 120         # 2 hour minimum — reduce fee-heavy flips
REENTRY_COOLDOWN_MINUTES = 60  # 60 min cooldown — avoid overtrading

# Fees — Coinbase Advanced Trade (DETECTED: 0.4% maker / 0.8% taker)
# Real detected tier: higher than expected
# STRATEGY: ALL orders are post_only limit = minimize fees. NEVER market orders.
MAKER_FEE_PCT = 0.004          # 0.40% — detected actual maker rate
TAKER_FEE_PCT = 0.008          # 0.80% — detected actual taker rate
ROUND_TRIP_FEE_PCT = 0.008     # 0.80% round trip (limit both sides)
ROUND_TRIP_FEE_TAKER_PCT = 0.012   # 1.20% if forced to use market exit
FEE_AUTO_DETECT = True         # Override with detected rates from API - CRITICAL

# Safety — ENFORCED
MAX_DRAWDOWN_PCT = 0.40         # 40% drawdown halt
DAILY_MAX_LOSS_PCT = 0.10       # 10% daily loss limit
MAX_CONSECUTIVE_LOSSES = 5      # 5 losses = 1 hour pause
COOLDOWN_AFTER_STOP_SEC = 30

# State
STATE_FILE = PROJECT_DIR / "state.json"

# AI — Dual Brain
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4-0709")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# TradingView
TV_ENABLED = True

# News
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")

# Analysis interval — 10 min optimized for workflow schedule
ANALYSIS_INTERVAL_SEC = 600

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

# ─── HYPERLIQUID ──────────────────────────────────────────────────
HYPERLIQUID_ENABLED = True
HYPERLIQUID_WALLET = {
    "address": os.environ.get("HL_WALLET_ADDRESS", "0x9FD699534Bd56c378B46eE60d212949F6Ea7A4d6"),
    "private_key": os.environ.get("HL_PRIVATE_KEY", ""),
}
HYPERLIQUID_DEFAULT_LEVERAGE = 3
HYPERLIQUID_MAX_LEVERAGE = 10

# Hyperliquid trading pairs (perps)
HL_TRADING_PAIRS = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK",
    "PEPE", "SUI", "NEAR", "RENDER", "FET",
    "INJ", "TIA", "SEI", "WIF", "ARB", "OP",
]

# Hyperliquid fees (way cheaper than Coinbase)
HL_MAKER_FEE_PCT = 0.00015     # 0.015%
HL_TAKER_FEE_PCT = 0.00045     # 0.045%
HL_ROUND_TRIP_MAKER = 0.0003   # 0.03% round trip (limit both sides)
HL_ROUND_TRIP_TAKER = 0.0006   # 0.06% round trip (limit entry + market exit)

# Hyperliquid risk params (can be more aggressive due to lower fees)
HL_MAX_POSITION_PCT = 0.30     # 30% per trade
HL_MIN_EXPECTED_MOVE_PCT = 0.5 # 0.5% min (vs 2.5% on Coinbase) — fees are 40x lower
HL_MIN_CONFIDENCE = 0.45       # Lower bar — fees don't eat you alive
