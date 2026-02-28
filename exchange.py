"""Coinbase Advanced Trade API wrapper — v2 with limit order support.

Real money only. No paper trading. No withdraw capability.
Uses limit orders (0.4% maker) instead of market orders (0.6% taker).
"""
import base64
import json
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import requests as http_requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

import config

log = logging.getLogger("exchange")

API_BASE = "https://api.coinbase.com/api/v3/brokerage"


class CoinbaseAuth:
    """Custom auth using EdDSA JWT (CDP portal keys use Ed25519, not ES256)."""

    def __init__(self):
        key_data = json.loads(config.COINBASE_KEY_FILE.read_text())
        self.key_id = key_data.get("name") or key_data.get("id", "")
        raw_pk = key_data.get("privateKey", "")

        # Convert raw base64 to Ed25519 PEM if needed
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
        """Build EdDSA JWT for API request."""
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
        """Make authenticated API request with 3-retry exponential backoff."""
        for attempt in range(3):
            try:
                token = self.build_jwt(method, path)
                url = f"https://api.coinbase.com{path}"
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }
                r = http_requests.request(
                    method, url, headers=headers,
                    params=params, json=json_data, timeout=15,
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    log.warning(f"API request {method} {path} failed (attempt {attempt+1}): {e} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    log.error(f"API request {method} {path} failed after 3 attempts: {e}")
                    raise

    def get(self, path, params=None):
        return self.request("GET", path, params=params)

    def post(self, path, json_data=None):
        return self.request("POST", path, json_data=json_data)

    def delete(self, path, params=None):
        return self.request("DELETE", path, params=params)


def get_auth():
    """Get authenticated Coinbase client."""
    if config.COINBASE_KEY_FILE.exists():
        return CoinbaseAuth()
    # Fallback: check for COINBASE_KEY_JSON env var (GitHub Actions)
    env_key = __import__("os").environ.get("COINBASE_KEY_JSON", "")
    if env_key:
        config.COINBASE_KEY_FILE.write_text(env_key)
        return CoinbaseAuth()
    raise RuntimeError("No coinbase_key.json found and COINBASE_KEY_JSON env var not set")


class Exchange:
    """Coinbase Advanced Trade — real money, limit orders, no paper mode."""

    def __init__(self):
        try:
            self.auth = get_auth()
        except Exception as e:
            log.error(f"Auth init failed: {e}")
            raise

    # ─── Account ──────────────────────────────────────────────────────

    def get_balances(self):
        """Return dict of {currency: available_balance}."""
        data = self.auth.get("/api/v3/brokerage/accounts")
        balances = {}
        for acct in data.get("accounts", []):
            bal = float(acct.get("available_balance", {}).get("value", 0))
            cur = acct.get("available_balance", {}).get("currency", "")
            if bal > 0 and cur:
                balances[cur] = bal
        return balances

    def get_total_usd_value(self):
        """Get total portfolio value in USD. Returns None if any price fetch fails."""
        balances = self.get_balances()
        total = balances.get("USD", 0.0)
        for cur, amt in balances.items():
            if cur == "USD":
                continue
            pair = f"{cur}-USD"
            price = self.get_price(pair)
            if price and price > 0:
                total += amt * price
            else:
                log.warning(f"Price fetch failed for {pair} (holding {amt:.6f}) — skipping cycle")
                return None
        return total

    # ─── Market Data ──────────────────────────────────────────────────

    def get_price(self, pair):
        """Get current price for a trading pair."""
        return self._get_live_price(pair)

    def _get_live_price(self, pair):
        """Get live price via public endpoint with retry."""
        for attempt in range(3):
            try:
                r = http_requests.get(
                    f"https://api.coinbase.com/api/v3/brokerage/market/products/{pair}",
                    timeout=10,
                )
                data = r.json()
                price = float(data.get("price", 0))
                if price > 0:
                    return price
            except Exception:
                pass
            # Fallback: v2 API
            try:
                base = pair.split("-")[0]
                r = http_requests.get(
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
        log.error(f"Price fetch failed for {pair} after 3 attempts")
        return None

    def get_candles(self, pair, granularity="ONE_HOUR", limit=100):
        """Get OHLCV candles. Returns list of dicts with open/high/low/close/volume/time."""
        end = datetime.now(timezone.utc)
        gran_seconds = {
            "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600, "SIX_HOUR": 21600, "ONE_DAY": 86400,
        }
        secs = gran_seconds.get(granularity, 3600)
        start = end - timedelta(seconds=secs * limit)

        try:
            r = http_requests.get(
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
            log.error(f"Candle fetch error for {pair}: {e}")
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

    def get_best_bid_ask(self, pair):
        """Get best bid and ask prices."""
        try:
            resp = self.auth.get("/api/v3/brokerage/best_bid_ask", params={"product_ids": pair})
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

        # Fallback: estimate from price
        price = self.get_price(pair)
        if price:
            spread = price * 0.0001
            return {"bid": round(price - spread, 2), "ask": round(price + spread, 2)}
        return {"bid": None, "ask": None}

    def get_product_info(self, pair):
        """Get product details including base/quote increments for order sizing."""
        try:
            r = http_requests.get(
                f"https://api.coinbase.com/api/v3/brokerage/market/products/{pair}",
                timeout=10,
            )
            data = r.json()
            return {
                "base_increment": data.get("base_increment", "0.00000001"),
                "quote_increment": data.get("quote_increment", "0.01"),
                "base_min_size": data.get("base_min_size", "0.00000001"),
                "base_max_size": data.get("base_max_size", "1000000"),
                "price": float(data.get("price", 0)),
            }
        except Exception as e:
            log.error(f"Product info fetch error for {pair}: {e}")
            return None

    # ─── Order Sizing Helpers ─────────────────────────────────────────

    @staticmethod
    def _truncate_to_increment(value, increment):
        """Truncate a float to the precision of the given increment string.

        Example: _truncate_to_increment(0.123456789, "0.0001") -> "0.1234"
        """
        if "." in increment:
            decimals = len(increment.rstrip("0").split(".")[-1])
        else:
            decimals = 0
        # Truncate, don't round — avoids exceeding balance
        factor = 10 ** decimals
        truncated = int(value * factor) / factor
        return f"{truncated:.{decimals}f}"

    # ─── Market Orders (fallback) ─────────────────────────────────────

    def market_buy(self, pair, usd_amount):
        """Market buy with USD amount. Fallback when limit order isn't viable."""
        order_id = str(uuid.uuid4())
        try:
            resp = self.auth.post("/api/v3/brokerage/orders", json_data={
                "client_order_id": order_id,
                "product_id": pair,
                "side": "BUY",
                "order_configuration": {
                    "market_market_ioc": {"quote_size": str(round(usd_amount, 2))}
                },
            })
            log.info(f"MARKET BUY {pair} ${usd_amount:.2f} — order: {resp.get('success_response', {}).get('order_id', 'unknown')}")
            return resp
        except Exception as e:
            log.error(f"Market buy failed for {pair}: {e}")
            return None

    def market_sell(self, pair, base_amount):
        """Market sell with base currency amount. Fallback when limit order isn't viable."""
        order_id = str(uuid.uuid4())
        product_info = self.get_product_info(pair)
        base_increment = product_info["base_increment"] if product_info else "0.00000001"
        base_size_str = self._truncate_to_increment(base_amount, base_increment)

        try:
            resp = self.auth.post("/api/v3/brokerage/orders", json_data={
                "client_order_id": order_id,
                "product_id": pair,
                "side": "SELL",
                "order_configuration": {
                    "market_market_ioc": {"base_size": base_size_str}
                },
            })
            log.info(f"MARKET SELL {pair} {base_size_str} — order: {resp.get('success_response', {}).get('order_id', 'unknown')}")
            return resp
        except Exception as e:
            log.error(f"Market sell failed for {pair}: {e}")
            return None

    # ─── Limit Orders (primary — 0.4% maker fee) ─────────────────────

    def limit_buy(self, pair, usd_amount, limit_price):
        """Place a limit buy order (post_only for maker fee).

        Args:
            pair: Trading pair e.g. "BTC-USD"
            usd_amount: USD value to spend
            limit_price: Maximum price to pay per unit

        Returns:
            API response dict or None on failure
        """
        order_id = str(uuid.uuid4())
        product_info = self.get_product_info(pair)
        if not product_info:
            log.error(f"Cannot place limit buy — product info unavailable for {pair}")
            return None

        base_increment = product_info["base_increment"]
        quote_increment = product_info["quote_increment"]

        # Calculate base size from USD amount and limit price
        base_size = usd_amount / limit_price
        base_size_str = self._truncate_to_increment(base_size, base_increment)
        limit_price_str = self._truncate_to_increment(limit_price, quote_increment)

        # Sanity check: don't place absurdly small orders
        if float(base_size_str) <= 0:
            log.error(f"Limit buy size too small after truncation: {base_size} -> {base_size_str}")
            return None

        try:
            resp = self.auth.post("/api/v3/brokerage/orders", json_data={
                "client_order_id": order_id,
                "product_id": pair,
                "side": "BUY",
                "order_configuration": {
                    "limit_limit_gtc": {
                        "base_size": base_size_str,
                        "limit_price": limit_price_str,
                        "post_only": True,
                    }
                },
            })
            oid = resp.get("success_response", {}).get("order_id", "unknown")
            log.info(f"LIMIT BUY {pair} {base_size_str} @ ${limit_price_str} (${usd_amount:.2f}) — order: {oid}")
            return resp
        except Exception as e:
            log.error(f"Limit buy failed for {pair}: {e}")
            return None

    def limit_sell(self, pair, base_amount, limit_price):
        """Place a limit sell order (post_only for maker fee).

        Args:
            pair: Trading pair e.g. "BTC-USD"
            base_amount: Amount of base currency to sell
            limit_price: Minimum price to accept per unit

        Returns:
            API response dict or None on failure
        """
        order_id = str(uuid.uuid4())
        product_info = self.get_product_info(pair)
        if not product_info:
            log.error(f"Cannot place limit sell — product info unavailable for {pair}")
            return None

        base_increment = product_info["base_increment"]
        quote_increment = product_info["quote_increment"]

        base_size_str = self._truncate_to_increment(base_amount, base_increment)
        limit_price_str = self._truncate_to_increment(limit_price, quote_increment)

        if float(base_size_str) <= 0:
            log.error(f"Limit sell size too small after truncation: {base_amount} -> {base_size_str}")
            return None

        try:
            resp = self.auth.post("/api/v3/brokerage/orders", json_data={
                "client_order_id": order_id,
                "product_id": pair,
                "side": "SELL",
                "order_configuration": {
                    "limit_limit_gtc": {
                        "base_size": base_size_str,
                        "limit_price": limit_price_str,
                        "post_only": True,
                    }
                },
            })
            oid = resp.get("success_response", {}).get("order_id", "unknown")
            log.info(f"LIMIT SELL {pair} {base_size_str} @ ${limit_price_str} — order: {oid}")
            return resp
        except Exception as e:
            log.error(f"Limit sell failed for {pair}: {e}")
            return None

    def get_order_status(self, order_id):
        """Check the status of an order by its ID.

        Returns dict with:
            - status: PENDING/OPEN/FILLED/CANCELLED/EXPIRED/FAILED
            - filled_size: amount filled
            - filled_value: USD value filled
            - average_filled_price: average execution price
            - completion_percentage: 0-100
        """
        try:
            resp = self.auth.get(f"/api/v3/brokerage/orders/historical/{order_id}")
            order = resp.get("order", {})
            status = order.get("status", "UNKNOWN")
            filled_size = float(order.get("filled_size", 0))
            filled_value = float(order.get("filled_value", 0))
            total_size = float(order.get("order_configuration", {})
                              .get("limit_limit_gtc", {})
                              .get("base_size", 0) or
                              order.get("order_configuration", {})
                              .get("market_market_ioc", {})
                              .get("base_size", 0) or 1)

            avg_price = filled_value / filled_size if filled_size > 0 else 0
            completion = (filled_size / total_size * 100) if total_size > 0 else 0

            return {
                "order_id": order_id,
                "status": status,
                "side": order.get("side", ""),
                "product_id": order.get("product_id", ""),
                "filled_size": filled_size,
                "filled_value": filled_value,
                "average_filled_price": round(avg_price, 2),
                "completion_percentage": round(min(completion, 100), 1),
                "created_time": order.get("created_time", ""),
                "last_fill_time": order.get("last_fill_time", ""),
            }
        except Exception as e:
            log.error(f"Order status check failed for {order_id}: {e}")
            return None

    def cancel_order(self, order_id):
        """Cancel an unfilled or partially filled order.

        Returns True if cancellation was successful, False otherwise.
        """
        try:
            resp = self.auth.post("/api/v3/brokerage/orders/batch_cancel", json_data={
                "order_ids": [order_id],
            })
            results = resp.get("results", [])
            if results:
                success = results[0].get("success", False)
                if success:
                    log.info(f"Order {order_id} cancelled successfully")
                else:
                    reason = results[0].get("failure_reason", "unknown")
                    log.warning(f"Order {order_id} cancel failed: {reason}")
                return success
            return False
        except Exception as e:
            log.error(f"Cancel order failed for {order_id}: {e}")
            return False

    def get_open_orders(self, pair=None):
        """Get all open orders, optionally filtered by pair."""
        try:
            params = {"order_status": "OPEN"}
            if pair:
                params["product_id"] = pair
            resp = self.auth.get("/api/v3/brokerage/orders/historical/batch", params=params)
            return resp.get("orders", [])
        except Exception as e:
            log.error(f"Get open orders error: {e}")
            return []

    def cancel_all_orders(self, pair=None):
        """Cancel all open orders, optionally filtered by pair.

        Returns number of orders successfully cancelled.
        """
        open_orders = self.get_open_orders(pair)
        if not open_orders:
            return 0

        order_ids = [o.get("order_id", "") for o in open_orders if o.get("order_id")]
        if not order_ids:
            return 0

        try:
            resp = self.auth.post("/api/v3/brokerage/orders/batch_cancel", json_data={
                "order_ids": order_ids,
            })
            results = resp.get("results", [])
            cancelled = sum(1 for r in results if r.get("success", False))
            log.info(f"Cancelled {cancelled}/{len(order_ids)} open orders" +
                     (f" for {pair}" if pair else ""))
            return cancelled
        except Exception as e:
            log.error(f"Cancel all orders failed: {e}")
            return 0

    def get_fees(self):
        """Get current fee tier."""
        try:
            resp = self.auth.get("/api/v3/brokerage/transaction_summary")
            return {
                "maker": float(resp.get("maker_fee_rate", 0.004)),
                "taker": float(resp.get("taker_fee_rate", 0.006)),
            }
        except Exception as e:
            log.error(f"Fee fetch error: {e}")
            return {"maker": config.MAKER_FEE_PCT, "taker": config.TAKER_FEE_PCT}


def verify_no_withdrawal_capability():
    """Safety check: scan the Exchange class to confirm no withdraw methods exist.

    This runs at import time as a safeguard. The Coinbase Advanced Trade API
    keys we use do NOT have withdrawal permissions, but this is defense in depth.
    """
    forbidden_patterns = ["withdraw", "send_money", "transfer_out", "payout"]
    methods = [m for m in dir(Exchange) if not m.startswith("_")]
    for method_name in methods:
        for pattern in forbidden_patterns:
            if pattern in method_name.lower():
                raise RuntimeError(
                    f"SAFETY VIOLATION: Exchange class has forbidden method '{method_name}' "
                    f"matching pattern '{pattern}'. Withdrawals are NOT allowed."
                )
    log.info("Safety check passed: no withdrawal methods found in Exchange class")


# Run safety check at module load
verify_no_withdrawal_capability()
