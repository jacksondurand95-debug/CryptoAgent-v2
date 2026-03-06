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
MAX_POSITION_PCT = 0.25        # 25% per trade (~$350) — controlled sizing
MAX_OPEN_POSITIONS = 3         # 3 concurrent — concentrate on best setups
STOP_LOSS_ATR_MULT = 2.0      # 2x ATR stops — give trades room to breathe
TRAILING_STOP_ATR_MULT = 1.8  # 1.8x ATR trailing — lock gains
TRAILING_ACTIVATION_R = 1.5   # Activate trailing after 1.5x risk achieved
TAKE_PROFIT_ATR_MULT = 8.0    # 8x ATR target — let winners run far
MIN_CONFIDENCE = 0.55          # Higher bar — quality over quantity
MIN_EXPECTED_MOVE_PCT = 3.0    # 4% min — must clear 3x fees to be worth it
TIME_STOP_HOURS = 48           # 2 days max hold — rotate capital
MIN_HOLD_MINUTES = 60          # 60 min minimum — stop cutting winners early
REENTRY_COOLDOWN_MINUTES = 30  # 30 min cooldown — avoid revenge trading

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
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
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
