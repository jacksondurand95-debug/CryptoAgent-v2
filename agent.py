#!/usr/bin/env python3
"""CryptoAgent v2.1 — Proven Algorithm Serverless Trading Agent.

Runs as a single shot from GitHub Actions every 10 minutes.
State persisted to state.json (committed by Actions workflow).

Based on:
- AdaptiveTrend (arXiv 2602.11708): Sharpe 2.41, 6H timeframe
- Bollinger-Keltner Squeeze: fewer but bigger trades
- Multi-source intel sub-agents for confidence overlay
"""
import base64
import json
import logging
import os
import secrets
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

import config
from indicators import compute_all
from strategies import analyze
from portfolio import (
    new_state, open_position, close_position, check_stops,
    has_position, can_reenter, get_stats,
)

# Intel sub-agent integration
try:
    from intel.aggregator import get_intel_brief
    INTEL_AVAILABLE = True
except ImportError:
    INTEL_AVAILABLE = False

# ─── LOGGING ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("agent")


# ─── COINBASE AUTH ────────────────────────────────────────────────

class CoinbaseAuth:
    """EdDSA JWT auth for Coinbase Advanced Trade API."""

    def __init__(self):
        key_data = json.loads(config.COINBASE_KEY_FILE.read_text())
        self.key_id = key_data.get("name") or key_data.get("id", "")
        raw_pk = key_data.get("privateKey", "")

        if raw_pk and "BEGIN" not in raw_pk:
            raw_bytes = base64.b64decode(raw_pk)
            ed_key = Ed25519PrivateKey.from_private_bytes(raw_bytes[:32])
            self.pem = ed_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        else:
            self.pem = raw_pk.encode()

    def build_jwt(self, method, path):
        uri = f"{method} api.coinbase.com{path}"
        jwt_data = {
            "sub": self.key_id,
            "iss": "cdp",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
            "uri": uri,
        }
        return pyjwt.encode(
            jwt_data, self.pem, algorithm="EdDSA",
            headers={"kid": self.key_id, "nonce": secrets.token_hex()},
        )

    def request(self, method, path, params=None, json_data=None):
        token = self.build_jwt(method, path)
        url = f"https://api.coinbase.com{path}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        for attempt in range(3):
            try:
                r = requests.request(method, url, headers=headers, params=params, json=json_data, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def get(self, path, params=None):
        return self.request("GET", path, params=params)

    def post(self, path, json_data=None):
        return self.request("POST", path, json_data=json_data)


# ─── FILE-BASED STATE PERSISTENCE ────────────────────────────────

STATE_FILE = config.PROJECT_DIR / "state.json"


def load_state():
    """Load state from state.json in the repo."""
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            log.info(f"Loaded state: {len(state.get('positions', []))} positions, "
                     f"{len(state.get('trades', []))} trades")
            for key in ("positions", "trades", "pending_orders"):
                state.setdefault(key, [])
            state.setdefault("starting_value", None)
            state.setdefault("peak_value", None)
            return state
        except Exception as e:
            log.warning(f"Failed to load state: {e} — starting fresh")
    return new_state()


def save_state(state):
    """Save state to state.json in the repo (committed by Actions workflow)."""
    if "trades" in state:
        state["trades"] = state["trades"][-200:]
    state["last_updated"] = time.time()
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
        log.info("State saved to state.json")
    except Exception as e:
        log.error(f"Failed to save state: {e}")


# ─── EXCHANGE FUNCTIONS ───────────────────────────────────────────

def get_price(pair):
    """Get current spot price via public endpoint (no auth needed)."""
    for attempt in range(3):
        try:
            r = requests.get(
                f"https://api.coinbase.com/api/v3/brokerage/market/products/{pair}",
                timeout=10,
            )
            data = r.json()
            price = float(data.get("price", 0))
            if price > 0:
                return price
        except Exception:
            pass
        try:
            base = pair.split("-")[0]
            r = requests.get(
                f"https://api.coinbase.com/v2/prices/{base}-USD/spot",
                timeout=10,
            )
            data = r.json()
            price = float(data.get("data", {}).get("amount", 0))
            if price > 0:
                return price
        except Exception:
            pass
        if attempt < 2:
            time.sleep(2 ** attempt)
    return None


def get_candles(pair, granularity="SIX_HOUR", limit=100):
    """Get OHLCV candles from Coinbase public endpoint."""
    end = datetime.now(timezone.utc)
    gran_seconds = {
        "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
        "ONE_HOUR": 3600, "SIX_HOUR": 21600, "ONE_DAY": 86400,
    }
    secs = gran_seconds.get(granularity, 21600)
    start = end - timedelta(seconds=secs * limit)

    try:
        r = requests.get(
            f"https://api.coinbase.com/api/v3/brokerage/market/products/{pair}/candles",
            params={
                "start": str(int(start.timestamp())),
                "end": str(int(end.timestamp())),
                "granularity": granularity,
            },
            timeout=15,
        )
        resp = r.json()
    except Exception as e:
        log.error(f"Candle fetch error {pair} {granularity}: {e}")
        return []

    candles = []
    raw = resp.get("candles", []) if isinstance(resp, dict) else resp
    for c in raw:
        candles.append({
            "time": int(c.get("start", 0)),
            "open": float(c.get("open", 0)),
            "high": float(c.get("high", 0)),
            "low": float(c.get("low", 0)),
            "close": float(c.get("close", 0)),
            "volume": float(c.get("volume", 0)),
        })
    candles.sort(key=lambda x: x["time"])
    return candles


def get_balances(auth):
    """Get account balances."""
    data = auth.get("/api/v3/brokerage/accounts")
    balances = {}
    for acct in data.get("accounts", []):
        bal = float(acct.get("available_balance", {}).get("value", 0))
        cur = acct.get("available_balance", {}).get("currency", "")
        if bal > 0 and cur:
            balances[cur] = bal
    return balances


def get_total_usd_value(auth):
    """Get total portfolio value in USD."""
    balances = get_balances(auth)
    total = balances.get("USD", 0.0)
    for cur, amt in balances.items():
        if cur == "USD":
            continue
        price = get_price(f"{cur}-USD")
        if price and price > 0:
            total += amt * price
        else:
            log.warning(f"Price fetch failed for {cur}-USD")
            return None
    return total


def get_best_bid_ask(auth, pair):
    """Get best bid/ask for limit orders."""
    try:
        resp = auth.get("/api/v3/brokerage/best_bid_ask", params={"product_ids": pair})
        pricebooks = resp.get("pricebooks", [])
        if pricebooks:
            pb = pricebooks[0]
            bids = pb.get("bids", [])
            asks = pb.get("asks", [])
            return {
                "bid": float(bids[0]["price"]) if bids else None,
                "ask": float(asks[0]["price"]) if asks else None,
            }
    except Exception:
        pass

    price = get_price(pair)
    if price:
        spread = price * 0.0001
        return {"bid": round(price - spread, 2), "ask": round(price + spread, 2)}
    return {"bid": None, "ask": None}


def place_limit_order(auth, pair, side, price, usd_amount):
    """Place a post_only limit order (guaranteed maker fee). Returns order dict or None."""
    order_id = str(uuid.uuid4())

    qty = usd_amount / price
    order_config = {
        "limit_limit_gtc": {
            "base_size": str(round(qty, 8)),
            "limit_price": str(round(price, 2)),
            "post_only": True,  # MAKER FEE GUARANTEED — reject if would be taker
        }
    }

    try:
        resp = auth.post("/api/v3/brokerage/orders", json_data={
            "client_order_id": order_id,
            "product_id": pair,
            "side": side,
            "order_configuration": order_config,
        })
        log.info(f"LIMIT {side} order placed: {pair} @ ${price:,.2f} ({usd_amount:.2f} USD) [post_only]")
        return {
            "id": resp.get("success_response", {}).get("order_id", order_id),
            "client_order_id": order_id,
            "pair": pair,
            "side": side.lower(),
            "type": "limit",
            "price": price,
            "qty": qty,
            "usd": usd_amount,
            "placed_at": time.time(),
            "status": "pending",
        }
    except Exception as e:
        log.error(f"Limit order failed: {e}")
        return None


def place_market_order(auth, pair, side, usd_amount=None, base_amount=None):
    """Place a market order as fallback. Returns order dict or None."""
    order_id = str(uuid.uuid4())

    if side == "BUY":
        order_config = {"market_market_ioc": {"quote_size": str(round(usd_amount, 2))}}
    else:
        order_config = {"market_market_ioc": {"base_size": str(round(base_amount, 8))}}

    try:
        resp = auth.post("/api/v3/brokerage/orders", json_data={
            "client_order_id": order_id,
            "product_id": pair,
            "side": side,
            "order_configuration": order_config,
        })
        log.info(f"MARKET {side} order placed: {pair}")
        return resp
    except Exception as e:
        log.error(f"Market order failed: {e}")
        return None


def check_order_status(auth, order_id):
    """Check if a pending order has been filled."""
    try:
        resp = auth.get(f"/api/v3/brokerage/orders/historical/{order_id}")
        order = resp.get("order", {})
        return order.get("status", "UNKNOWN")
    except Exception as e:
        log.warning(f"Order status check failed: {e}")
        return "UNKNOWN"


def cancel_order(auth, order_id):
    """Cancel a pending order."""
    try:
        auth.post("/api/v3/brokerage/orders/batch_cancel", json_data={"order_ids": [order_id]})
        log.info(f"Cancelled order {order_id}")
        return True
    except Exception as e:
        log.warning(f"Cancel failed: {e}")
        return False


# ─── DATA COLLECTION ──────────────────────────────────────────────

def fetch_okx_data(pair):
    """Fetch OKX derivatives data (funding, OI, long/short, taker volume)."""
    base = pair.split("-")[0]
    inst_id = f"{base}-USDT-SWAP"
    data = {}

    # Funding rate
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": inst_id},
            timeout=10,
        )
        resp = r.json()
        if resp.get("data"):
            rate = float(resp["data"][0].get("fundingRate", 0))
            next_rate = float(resp["data"][0].get("nextFundingRate", 0))
            data["funding"] = {
                "current": rate * 100,
                "next": next_rate * 100,
                "signal": {
                    "bias": "bullish" if rate < -0.0001 else "bearish" if rate > 0.0003 else "neutral",
                    "strength": "strong" if abs(rate) > 0.0005 else "moderate" if abs(rate) > 0.0002 else "weak",
                },
                "negative_streak": 0,
                "positive_streak": 0,
            }
    except Exception as e:
        log.debug(f"OKX funding error: {e}")

    # Open interest
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/public/open-interest",
            params={"instType": "SWAP", "instId": inst_id},
            timeout=10,
        )
        resp = r.json()
        if resp.get("data"):
            oi_val = float(resp["data"][0].get("oi", 0))
            data["open_interest"] = {
                "current": oi_val,
                "change_2h_pct": 0,
                "change_4h_pct": 0,
            }
    except Exception as e:
        log.debug(f"OKX OI error: {e}")

    # Long/Short ratio (global)
    try:
        r = requests.get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio/{base}",
            params={"period": "1H"},
            timeout=10,
        )
        resp = r.json()
        if resp.get("data"):
            latest = resp["data"][0]
            ratio = float(latest[1]) if len(latest) > 1 else 1.0
            data["long_short_ratio"] = {
                "current": ratio,
                "extreme_short": ratio < 0.7,
                "extreme_long": ratio > 1.5,
            }
    except Exception as e:
        log.debug(f"OKX L/S error: {e}")

    # Top trader position ratio
    try:
        r = requests.get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader/{base}",
            params={"period": "1H"},
            timeout=10,
        )
        resp = r.json()
        if resp.get("data"):
            latest = resp["data"][0]
            top_ratio = float(latest[1]) if len(latest) > 1 else 1.0
            data["top_traders"] = {
                "ratio": top_ratio,
                "whales_long": top_ratio > 1.3,
                "whales_short": top_ratio < 0.7,
            }
    except Exception as e:
        log.debug(f"OKX top traders error: {e}")

    # Taker buy/sell volume
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/rubik/stat/taker-volume",
            params={"ccy": base, "instType": "SWAP", "period": "1H"},
            timeout=10,
        )
        resp = r.json()
        if resp.get("data"):
            latest = resp["data"][0]
            buy_vol = float(latest[1]) if len(latest) > 1 else 0
            sell_vol = float(latest[2]) if len(latest) > 2 else 0
            data["taker_buy_vol"] = buy_vol
            data["taker_sell_vol"] = sell_vol
            ratio = buy_vol / sell_vol if sell_vol > 0 else 1.0
            data["taker_ratio"] = {
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "ratio": ratio,
                "aggressive_buyers": ratio > 1.1,
            }
    except Exception as e:
        log.debug(f"OKX taker error: {e}")

    return data


def fetch_fear_greed():
    """Fetch Fear & Greed Index (free, no auth)."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()
        if data.get("data"):
            val = int(data["data"][0].get("value", 50))
            classification = data["data"][0].get("value_classification", "Neutral")
            return {"value": val, "classification": classification}
    except Exception:
        pass
    return None


def fetch_tradingview_analysis(pair):
    """Fetch TradingView technical analysis summary."""
    try:
        from tradingview_ta import TA_Handler, Interval

        base = pair.split("-")[0]
        tv_map = {
            "BTC": ("BTCUSD", "COINBASE"),
            "ETH": ("ETHUSD", "COINBASE"),
            "SOL": ("SOLUSD", "COINBASE"),
        }
        symbol, exchange = tv_map.get(base, (f"{base}USD", "COINBASE"))

        handler = TA_Handler(
            symbol=symbol,
            screener="crypto",
            exchange=exchange,
            interval=Interval.INTERVAL_4_HOURS,  # 4H for better signal quality
        )
        analysis = handler.get_analysis()
        summary = analysis.summary

        return {
            "RECOMMENDATION": summary.get("RECOMMENDATION", "NEUTRAL"),
            "BUY": summary.get("BUY", 0),
            "SELL": summary.get("SELL", 0),
            "NEUTRAL": summary.get("NEUTRAL", 0),
        }
    except Exception as e:
        log.debug(f"TradingView error for {pair}: {e}")
        return None


# ─── PENDING ORDER MANAGEMENT ────────────────────────────────────

def process_pending_orders(state, auth):
    """Check pending limit orders from previous runs.

    If filled: update positions.
    If pending for > 30 min: cancel (wider window for bigger moves).
    """
    if "pending_orders" not in state:
        state["pending_orders"] = []

    still_pending = []
    now = time.time()

    for order in state.get("pending_orders", []):
        order_id = order.get("id", "")
        if not order_id:
            continue

        status = check_order_status(auth, order_id)
        age_minutes = (now - order.get("placed_at", now)) / 60

        if status in ("FILLED", "COMPLETED"):
            log.info(f"ORDER FILLED: {order['side']} {order['pair']} @ ${order['price']:,.2f}")
            pair = order["pair"]
            price = order["price"]
            qty = order.get("qty", 0)
            usd = order.get("usd", 0)

            if order["side"] == "buy":
                atr_est = order.get("atr") or price * 0.025
                stop = order.get("stop_loss", round(price - atr_est * config.STOP_LOSS_ATR_MULT, 2))
                target = order.get("take_profit", round(price + atr_est * config.TAKE_PROFIT_ATR_MULT, 2))
                state = open_position(
                    state, pair, "long", price, qty, usd, stop, target, atr=atr_est
                )

        elif status in ("CANCELLED", "EXPIRED", "FAILED"):
            log.info(f"ORDER {status}: {order['pair']} — removing")

        elif age_minutes > 30:
            # Post_only orders that haven't filled in 30 min — cancel
            log.info(f"ORDER STALE ({age_minutes:.0f}min): cancelling {order['pair']}")
            cancel_order(auth, order_id)

        else:
            still_pending.append(order)
            log.info(f"ORDER PENDING: {order['pair']} ({age_minutes:.0f}min old)")

    state["pending_orders"] = still_pending
    return state


# ─── MAIN AGENT ───────────────────────────────────────────────────

def run():
    """Single-shot agent execution."""
    start_time = time.time()
    log.info("=" * 60)
    log.info("CryptoAgent v2.1 — Proven Algorithms (AdaptiveTrend/Squeeze)")
    log.info("=" * 60)

    # 1. Load state
    state = load_state()

    # 2. Initialize exchange auth
    try:
        auth = CoinbaseAuth()
        log.info("Coinbase auth initialized")
    except Exception as e:
        log.error(f"Auth failed: {e}")
        save_state(state)
        return

    # 3. Get portfolio value
    total_value = get_total_usd_value(auth)
    if total_value is None:
        log.error("Cannot get portfolio value — skipping run")
        save_state(state)
        return

    if state.get("starting_value") is None:
        state["starting_value"] = total_value
        log.info(f"First run — starting value: ${total_value:,.2f}")

    log.info(f"Portfolio value: ${total_value:,.2f}")

    # 4. Check pending orders from previous runs
    state = process_pending_orders(state, auth)

    # 5. Check stops on existing positions
    def _get_price(pair):
        return get_price(pair)

    state, stopped_trades = check_stops(state, _get_price)
    for trade in stopped_trades:
        sign = "+" if trade["pnl_usd"] >= 0 else ""
        log.info(
            f"EXIT: {trade['pair']} {trade['reason']} | "
            f"P&L: {sign}${trade['pnl_usd']:,.2f} ({sign}{trade['pnl_pct']:.1f}%)"
        )
        if trade["side"] == "long":
            try:
                place_market_order(auth, trade["pair"], "SELL", base_amount=trade["qty"])
            except Exception as e:
                log.error(f"Exit sell failed for {trade['pair']}: {e}")

    # 5b. Load intel sub-agent data
    intel_brief = None
    if INTEL_AVAILABLE:
        try:
            intel_brief = get_intel_brief()
            agg = intel_brief.get("aggregate", {})
            log.info(f"INTEL: score={agg.get('score', 0)} bias={agg.get('bias', 'N/A')} "
                     f"strength={agg.get('strength', 'N/A')} "
                     f"confidence={agg.get('confidence', 0):.0%} "
                     f"({agg.get('available_sources', 0)}/{agg.get('total_sources', 0)} sources)")
            for evt in intel_brief.get("alpha_events", [])[:3]:
                log.info(f"  ALPHA [{evt.get('importance')}]: {evt.get('title', '')[:70]}")
        except Exception as e:
            log.warning(f"Intel load failed (non-fatal): {e}")
    else:
        log.info("INTEL: sub-agents not available — running without intel overlay")

    # 5c. Fear & Greed Index
    fgi = fetch_fear_greed()
    if fgi:
        log.info(f"Fear & Greed: {fgi['value']} ({fgi['classification']})")

    # 6. Collect market data and run strategies for each pair
    signals_found = []
    actions_taken = []

    for pair in config.TRADING_PAIRS:
        log.info(f"--- {pair} ---")

        # Get candles at multiple timeframes (PRIMARY: 6H)
        candles_6h = get_candles(pair, config.PRIMARY_TIMEFRAME, 100)
        candles_1d = get_candles(pair, config.TREND_TIMEFRAME, 60)

        if not candles_6h or len(candles_6h) < 30:
            log.warning(f"  Insufficient 6H candle data for {pair} ({len(candles_6h) if candles_6h else 0})")
            continue

        # Compute indicators on 6H (primary) and 1D (trend filter)
        ind_6h = compute_all(candles_6h)
        ind_1d = compute_all(candles_1d) if candles_1d and len(candles_1d) >= 30 else {}

        if not ind_6h:
            log.warning(f"  No indicators for {pair}")
            continue

        price = ind_6h.get("price", 0)
        squeeze_str = f"SQ={ind_6h.get('squeeze_bars', 0)}bars" if ind_6h.get("squeeze") else "no-sq"
        mom_str = f"MOM28={ind_6h.get('momentum_28', 0):+.3f}"
        log.info(f"  Price: ${price:,.2f} | RSI={ind_6h.get('rsi', '?')} "
                 f"ADX={ind_6h.get('adx', '?')} {squeeze_str} {mom_str}")

        # Fetch on-chain data
        onchain = fetch_okx_data(pair)
        funding_str = ""
        if "funding" in onchain:
            funding_str = f" FR={onchain['funding']['current']:.4f}%"
        log.info(f"  OKX data: {len(onchain)} feeds{funding_str}")

        # Fetch TradingView analysis
        tv_analysis = None
        if config.TV_ENABLED:
            tv_analysis = fetch_tradingview_analysis(pair)
            if tv_analysis:
                log.info(f"  TV: {tv_analysis['RECOMMENDATION']} "
                         f"({tv_analysis['BUY']}B/{tv_analysis['SELL']}S/{tv_analysis['NEUTRAL']}N)")

        # Run ALL strategies (now uses 6H + 1D multi-timeframe)
        signal = analyze(pair, ind_6h, ind_1d, onchain, tv_analysis)

        if signal is None:
            log.info(f"  No signal for {pair}")
            continue

        # Apply intel overlay — boost or dampen confidence
        if intel_brief:
            coin = pair.split("-")[0]
            coin_intel = intel_brief.get("coins", {}).get(coin, {})
            intel_bias = coin_intel.get("bias", "neutral")
            master_bias = intel_brief.get("aggregate", {}).get("bias", "neutral")
            master_score = intel_brief.get("aggregate", {}).get("score", 0)

            original_conf = signal["confidence"]

            # Boost: signal aligns with intel
            if signal["action"] == "buy" and master_bias == "bullish":
                signal["confidence"] = min(signal["confidence"] + 0.05, 0.95)
            elif signal["action"] == "sell" and master_bias == "bearish":
                signal["confidence"] = min(signal["confidence"] + 0.05, 0.95)

            # Per-coin boost
            if signal["action"] == "buy" and intel_bias == "bullish":
                signal["confidence"] = min(signal["confidence"] + 0.03, 0.95)

            # Dampen: signal conflicts with strong intel
            if signal["action"] == "buy" and master_score < -50:
                signal["confidence"] *= 0.85
            elif signal["action"] == "sell" and master_score > 50:
                signal["confidence"] *= 0.85

            # Fear & Greed contrarian overlay
            if fgi:
                if signal["action"] == "buy" and fgi["value"] < 25:
                    signal["confidence"] = min(signal["confidence"] + 0.03, 0.95)
                elif signal["action"] == "buy" and fgi["value"] > 80:
                    signal["confidence"] *= 0.90

            if signal["confidence"] != original_conf:
                log.info(f"  OVERLAY adj: {original_conf:.2f} -> {signal['confidence']:.2f}")

        # Check minimum confidence (already filtered in analyze(), but double-check)
        if signal["confidence"] < config.MIN_CONFIDENCE:
            log.info(f"  Signal below min confidence: {signal['confidence']:.2f} < {config.MIN_CONFIDENCE}")
            continue

        signals_found.append(signal)

        # Check if we already have a position
        if signal["action"] == "buy" and has_position(state, pair):
            log.info(f"  Already have position in {pair} — skipping")
            continue

        # Check reentry cooldown
        if signal["action"] == "buy" and not can_reenter(state, pair):
            log.info(f"  Reentry cooldown active for {pair}")
            continue

        # Check max open positions
        if signal["action"] == "buy" and len(state["positions"]) >= config.MAX_OPEN_POSITIONS:
            log.info(f"  Max positions reached ({config.MAX_OPEN_POSITIONS})")
            continue

        # Check pending buy orders (don't double up)
        pending_pairs = {o["pair"] for o in state.get("pending_orders", []) if o["side"] == "buy"}
        if signal["action"] == "buy" and pair in pending_pairs:
            log.info(f"  Already have pending buy for {pair}")
            continue

        # 7. Execute signal
        if signal["action"] == "buy":
            size_pct = signal.get("size_pct", 25) / 100.0
            max_size = config.MAX_POSITION_PCT
            size_pct = min(size_pct, max_size)
            usd_amount = total_value * size_pct

            balances = get_balances(auth)
            available_usd = balances.get("USD", 0)
            if usd_amount > available_usd:
                usd_amount = available_usd * 0.95
            if usd_amount < 5:
                log.warning(f"  Insufficient USD: ${available_usd:.2f}")
                continue

            # Get best bid for limit order (maker fee with post_only!)
            bid_ask = get_best_bid_ask(auth, pair)
            limit_price = bid_ask.get("bid") or price

            order = place_limit_order(auth, pair, "BUY", limit_price, usd_amount)
            if order:
                order["stop_loss"] = signal.get("stop_loss")
                order["take_profit"] = signal.get("take_profit")
                order["atr"] = signal.get("atr")
                state.setdefault("pending_orders", []).append(order)
                actions_taken.append(
                    f"LIMIT BUY {pair} ${usd_amount:.2f} @ ${limit_price:,.2f} "
                    f"[{signal['strategy']}]"
                )

        elif signal["action"] == "sell":
            if has_position(state, pair):
                exit_price = get_price(pair)
                if exit_price:
                    for pos in state["positions"]:
                        if pos["pair"] == pair:
                            bid_ask = get_best_bid_ask(auth, pair)
                            limit_price = bid_ask.get("ask") or exit_price

                            order = place_limit_order(
                                auth, pair, "SELL", limit_price,
                                pos["qty"] * limit_price
                            )
                            if order:
                                order["base_qty"] = pos["qty"]
                                state.setdefault("pending_orders", []).append(order)

                            state, trade = close_position(
                                state, pair, exit_price, reason=signal["strategy"]
                            )
                            if trade:
                                actions_taken.append(
                                    f"SELL {pair} P&L: ${trade['pnl_usd']:+,.2f} [{signal['strategy']}]"
                                )
                            break

    # 8. Save state
    save_state(state)

    # 9. Print summary
    elapsed = time.time() - start_time
    pos_pairs = [p["pair"].split("-")[0] for p in state.get("positions", [])]
    pos_str = ", ".join(pos_pairs) if pos_pairs else "none"
    pending_count = len(state.get("pending_orders", []))

    best_signal_str = "none"
    if signals_found:
        best = signals_found[0]
        best_signal_str = (
            f"{best['action'].upper()} {best['pair'].split('-')[0]} "
            f"@ {best['confidence']:.2f} [{best['strategy']}]"
        )

    action_str = " | ".join(actions_taken) if actions_taken else "HOLD"

    stats = get_stats(state.get("trades", []))
    pnl_str = f"${stats.get('total_pnl_usd', 0):+,.2f}" if stats.get("total_trades", 0) > 0 else "$0"
    wr_str = f"{stats.get('win_rate', 0):.0f}%" if stats.get("total_trades", 0) > 0 else "N/A"

    log.info("")
    log.info("=" * 60)
    log.info(
        f"[AGENT] Portfolio: ${total_value:,.2f} | "
        f"Positions: {len(state.get('positions', []))} ({pos_str}) | "
        f"Signal: {best_signal_str} | "
        f"Action: {action_str}"
    )
    log.info(
        f"[STATS] Trades: {stats.get('total_trades', 0)} | "
        f"Win Rate: {wr_str} | "
        f"P&L: {pnl_str} | "
        f"Pending: {pending_count} | "
        f"Runtime: {elapsed:.1f}s"
    )
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log.critical(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
