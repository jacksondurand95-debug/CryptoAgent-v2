"""CryptoAgent v3.0 — Multi-Exchange Derivatives Data Feeds.

Aggregates derivatives intel from Bybit, OKX, and Binance.
All public endpoints — no API keys needed.
Provides: funding rates, open interest, liquidations, order book depth,
long/short ratios, whale positioning across ALL major exchanges.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger("data_feeds")

TIMEOUT = 8  # Fast timeout — skip slow endpoints, don't block


# ─── BYBIT PUBLIC API (v5) ────────────────────────────────────────

def fetch_bybit_funding(symbol="BTCUSDT"):
    """Bybit funding rate — current and historical."""
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/funding/history",
            params={"category": "linear", "symbol": symbol, "limit": "5"},
            timeout=TIMEOUT,
        )
        data = r.json()
        if data.get("retCode") != 0:
            return None
        rows = data.get("result", {}).get("list", [])
        if not rows:
            return None
        current = float(rows[0].get("fundingRate", 0))
        rates = [float(r.get("fundingRate", 0)) for r in rows]
        avg = sum(rates) / len(rates) if rates else 0
        return {
            "exchange": "bybit",
            "symbol": symbol,
            "current": current * 100,  # as percentage
            "avg_5": avg * 100,
            "trend": "rising" if len(rates) >= 2 and rates[0] > rates[1] else "falling",
            "extreme_positive": current > 0.0005,
            "extreme_negative": current < -0.0002,
        }
    except Exception as e:
        log.debug(f"Bybit funding error {symbol}: {e}")
        return None


def fetch_bybit_oi(symbol="BTCUSDT"):
    """Bybit open interest — current snapshot."""
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/open-interest",
            params={"category": "linear", "symbol": symbol, "intervalTime": "1h", "limit": "5"},
            timeout=TIMEOUT,
        )
        data = r.json()
        if data.get("retCode") != 0:
            return None
        rows = data.get("result", {}).get("list", [])
        if not rows:
            return None
        current_oi = float(rows[0].get("openInterest", 0))
        prev_oi = float(rows[-1].get("openInterest", 0)) if len(rows) > 1 else current_oi
        change_pct = ((current_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        return {
            "exchange": "bybit",
            "symbol": symbol,
            "current": current_oi,
            "change_pct": round(change_pct, 2),
            "rising": change_pct > 2,
            "falling": change_pct < -2,
        }
    except Exception as e:
        log.debug(f"Bybit OI error {symbol}: {e}")
        return None


def fetch_bybit_tickers(symbol="BTCUSDT"):
    """Bybit ticker — 24h volume, turnover, price changes."""
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=TIMEOUT,
        )
        data = r.json()
        if data.get("retCode") != 0:
            return None
        tickers = data.get("result", {}).get("list", [])
        if not tickers:
            return None
        t = tickers[0]
        return {
            "exchange": "bybit",
            "symbol": symbol,
            "last_price": float(t.get("lastPrice", 0)),
            "mark_price": float(t.get("markPrice", 0)),
            "index_price": float(t.get("indexPrice", 0)),
            "volume_24h": float(t.get("volume24h", 0)),
            "turnover_24h": float(t.get("turnover24h", 0)),
            "price_change_24h_pct": float(t.get("price24hPcnt", 0)) * 100,
            "funding_rate": float(t.get("fundingRate", 0)) * 100,
            "next_funding_time": t.get("nextFundingTime", ""),
            "open_interest": float(t.get("openInterest", 0)),
            "bid": float(t.get("bid1Price", 0)),
            "ask": float(t.get("ask1Price", 0)),
        }
    except Exception as e:
        log.debug(f"Bybit ticker error {symbol}: {e}")
        return None


def fetch_bybit_long_short(symbol="BTCUSDT"):
    """Bybit long/short ratio."""
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/account-ratio",
            params={"category": "linear", "symbol": symbol, "period": "1h", "limit": "3"},
            timeout=TIMEOUT,
        )
        data = r.json()
        if data.get("retCode") != 0:
            return None
        rows = data.get("result", {}).get("list", [])
        if not rows:
            return None
        buy_ratio = float(rows[0].get("buyRatio", 0.5))
        sell_ratio = float(rows[0].get("sellRatio", 0.5))
        ls_ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1.0
        return {
            "exchange": "bybit",
            "symbol": symbol,
            "long_pct": buy_ratio * 100,
            "short_pct": sell_ratio * 100,
            "ratio": round(ls_ratio, 3),
            "extreme_long": ls_ratio > 1.5,
            "extreme_short": ls_ratio < 0.7,
        }
    except Exception as e:
        log.debug(f"Bybit L/S error {symbol}: {e}")
        return None


# ─── BINANCE FUTURES PUBLIC API ───────────────────────────────────

def fetch_binance_funding(symbol="BTCUSDT"):
    """Binance funding rate."""
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 5},
            timeout=TIMEOUT,
        )
        rows = r.json()
        if not rows:
            return None
        current = float(rows[-1].get("fundingRate", 0))
        rates = [float(r.get("fundingRate", 0)) for r in rows]
        avg = sum(rates) / len(rates)
        return {
            "exchange": "binance",
            "symbol": symbol,
            "current": current * 100,
            "avg_5": avg * 100,
            "extreme_positive": current > 0.0005,
            "extreme_negative": current < -0.0002,
        }
    except Exception as e:
        log.debug(f"Binance funding error {symbol}: {e}")
        return None


def fetch_binance_oi(symbol="BTCUSDT"):
    """Binance open interest."""
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=TIMEOUT,
        )
        data = r.json()
        oi = float(data.get("openInterest", 0))
        return {
            "exchange": "binance",
            "symbol": symbol,
            "current": oi,
        }
    except Exception as e:
        log.debug(f"Binance OI error {symbol}: {e}")
        return None


def fetch_binance_long_short_global(symbol="BTCUSDT"):
    """Binance global long/short ratio."""
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": "1h", "limit": 3},
            timeout=TIMEOUT,
        )
        rows = r.json()
        if not rows:
            return None
        latest = rows[-1]
        ratio = float(latest.get("longShortRatio", 1.0))
        long_pct = float(latest.get("longAccount", 0.5)) * 100
        short_pct = float(latest.get("shortAccount", 0.5)) * 100
        return {
            "exchange": "binance",
            "symbol": symbol,
            "ratio": ratio,
            "long_pct": long_pct,
            "short_pct": short_pct,
            "extreme_long": ratio > 1.5,
            "extreme_short": ratio < 0.7,
        }
    except Exception as e:
        log.debug(f"Binance L/S error {symbol}: {e}")
        return None


def fetch_binance_top_trader_positions(symbol="BTCUSDT"):
    """Binance top trader long/short ratio (positions)."""
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
            params={"symbol": symbol, "period": "1h", "limit": 3},
            timeout=TIMEOUT,
        )
        rows = r.json()
        if not rows:
            return None
        latest = rows[-1]
        ratio = float(latest.get("longShortRatio", 1.0))
        return {
            "exchange": "binance",
            "symbol": symbol,
            "top_trader_ratio": ratio,
            "whales_long": ratio > 1.3,
            "whales_short": ratio < 0.7,
        }
    except Exception as e:
        log.debug(f"Binance top traders error {symbol}: {e}")
        return None


def fetch_binance_taker_volume(symbol="BTCUSDT"):
    """Binance taker buy/sell volume ratio."""
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/takerlongshortRatio",
            params={"symbol": symbol, "period": "1h", "limit": 3},
            timeout=TIMEOUT,
        )
        rows = r.json()
        if not rows:
            return None
        latest = rows[-1]
        buy_vol = float(latest.get("buyVol", 0))
        sell_vol = float(latest.get("sellVol", 0))
        ratio = buy_vol / sell_vol if sell_vol > 0 else 1.0
        return {
            "exchange": "binance",
            "symbol": symbol,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "ratio": round(ratio, 3),
            "aggressive_buyers": ratio > 1.1,
            "aggressive_sellers": ratio < 0.9,
        }
    except Exception as e:
        log.debug(f"Binance taker error {symbol}: {e}")
        return None


# ─── OKX PUBLIC API (v5) — FIXED ENDPOINTS ───────────────────────

def fetch_okx_funding(inst_id="BTC-USDT-SWAP"):
    """OKX funding rate."""
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": inst_id},
            timeout=TIMEOUT,
        )
        resp = r.json()
        if resp.get("data"):
            rate = float(resp["data"][0].get("fundingRate", 0))
            next_rate = float(resp["data"][0].get("nextFundingRate", 0))
            return {
                "exchange": "okx",
                "inst_id": inst_id,
                "current": rate * 100,
                "next": next_rate * 100,
                "extreme_positive": rate > 0.0005,
                "extreme_negative": rate < -0.0002,
            }
    except Exception as e:
        log.debug(f"OKX funding error {inst_id}: {e}")
    return None


def fetch_okx_oi(inst_id="BTC-USDT-SWAP"):
    """OKX open interest."""
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/public/open-interest",
            params={"instType": "SWAP", "instId": inst_id},
            timeout=TIMEOUT,
        )
        resp = r.json()
        if resp.get("data"):
            return {
                "exchange": "okx",
                "inst_id": inst_id,
                "current": float(resp["data"][0].get("oi", 0)),
            }
    except Exception as e:
        log.debug(f"OKX OI error {inst_id}: {e}")
    return None


# ─── AGGREGATED CROSS-EXCHANGE DATA ──────────────────────────────

PAIR_TO_SYMBOLS = {
    "BTC-USD": {"bybit": "BTCUSDT", "binance": "BTCUSDT", "okx": "BTC-USDT-SWAP"},
    "ETH-USD": {"bybit": "ETHUSDT", "binance": "ETHUSDT", "okx": "ETH-USDT-SWAP"},
    "SOL-USD": {"bybit": "SOLUSDT", "binance": "SOLUSDT", "okx": "SOL-USDT-SWAP"},
    "DOGE-USD": {"bybit": "DOGEUSDT", "binance": "DOGEUSDT", "okx": "DOGE-USDT-SWAP"},
    "AVAX-USD": {"bybit": "AVAXUSDT", "binance": "AVAXUSDT", "okx": "AVAX-USDT-SWAP"},
    "LINK-USD": {"bybit": "LINKUSDT", "binance": "LINKUSDT", "okx": "LINK-USDT-SWAP"},
}


def fetch_all_derivatives(pair):
    """Fetch aggregated derivatives data for a trading pair across all exchanges.

    Returns a rich dict with cross-exchange funding, OI, long/short, taker volume,
    and computed aggregate signals.
    """
    symbols = PAIR_TO_SYMBOLS.get(pair, {})
    if not symbols:
        return {}

    bybit_sym = symbols.get("bybit", "")
    binance_sym = symbols.get("binance", "")
    okx_inst = symbols.get("okx", "")

    results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=12) as pool:
        # Bybit
        if bybit_sym:
            futures["bybit_funding"] = pool.submit(fetch_bybit_funding, bybit_sym)
            futures["bybit_oi"] = pool.submit(fetch_bybit_oi, bybit_sym)
            futures["bybit_ticker"] = pool.submit(fetch_bybit_tickers, bybit_sym)
            futures["bybit_ls"] = pool.submit(fetch_bybit_long_short, bybit_sym)

        # Binance
        if binance_sym:
            futures["binance_funding"] = pool.submit(fetch_binance_funding, binance_sym)
            futures["binance_oi"] = pool.submit(fetch_binance_oi, binance_sym)
            futures["binance_ls_global"] = pool.submit(fetch_binance_long_short_global, binance_sym)
            futures["binance_top_traders"] = pool.submit(fetch_binance_top_trader_positions, binance_sym)
            futures["binance_taker"] = pool.submit(fetch_binance_taker_volume, binance_sym)

        # OKX
        if okx_inst:
            futures["okx_funding"] = pool.submit(fetch_okx_funding, okx_inst)
            futures["okx_oi"] = pool.submit(fetch_okx_oi, okx_inst)

        for key, future in futures.items():
            try:
                result = future.result(timeout=TIMEOUT + 2)
                if result:
                    results[key] = result
            except Exception as e:
                log.debug(f"Feed {key} failed for {pair}: {e}")

    # Compute aggregate signals
    agg = _compute_aggregates(results)
    results["aggregate"] = agg
    results["feed_count"] = len([k for k, v in results.items() if v and k != "aggregate"])

    return results


def _compute_aggregates(results):
    """Compute cross-exchange aggregate signals from individual feeds."""
    agg = {
        "funding_bias": "neutral",
        "funding_extreme": False,
        "oi_trend": "neutral",
        "positioning_bias": "neutral",
        "smart_money_signal": "neutral",
        "taker_flow": "neutral",
        "overall_bias": "neutral",
        "signal_count": 0,
        "bullish_signals": 0,
        "bearish_signals": 0,
    }

    bullish = 0
    bearish = 0
    total = 0

    # Aggregate funding rates
    funding_rates = []
    for key in ("bybit_funding", "binance_funding", "okx_funding"):
        f = results.get(key)
        if f:
            funding_rates.append(f["current"])

    if funding_rates:
        avg_funding = sum(funding_rates) / len(funding_rates)
        if avg_funding < -0.02:
            agg["funding_bias"] = "bullish"  # negative funding = short squeeze potential
            bullish += 2
        elif avg_funding > 0.05:
            agg["funding_bias"] = "bearish"  # overleveraged longs
            bearish += 2
        if any(abs(fr) > 0.05 for fr in funding_rates):
            agg["funding_extreme"] = True
        total += 2

    # Aggregate OI changes
    bybit_oi = results.get("bybit_oi")
    if bybit_oi:
        if bybit_oi.get("rising"):
            agg["oi_trend"] = "rising"
        elif bybit_oi.get("falling"):
            agg["oi_trend"] = "falling"
        total += 1

    # Aggregate long/short ratios
    ls_ratios = []
    for key in ("bybit_ls", "binance_ls_global"):
        ls = results.get(key)
        if ls:
            ls_ratios.append(ls["ratio"])
    if ls_ratios:
        avg_ls = sum(ls_ratios) / len(ls_ratios)
        if avg_ls > 1.3:
            agg["positioning_bias"] = "crowded_long"
            bearish += 1  # contrarian: crowded longs = bearish
        elif avg_ls < 0.75:
            agg["positioning_bias"] = "crowded_short"
            bullish += 1  # contrarian: crowded shorts = bullish
        total += 1

    # Smart money (top trader positioning)
    top = results.get("binance_top_traders")
    if top:
        if top.get("whales_long"):
            agg["smart_money_signal"] = "bullish"
            bullish += 2  # smart money has 2x weight
        elif top.get("whales_short"):
            agg["smart_money_signal"] = "bearish"
            bearish += 2
        total += 2

    # Taker flow
    taker = results.get("binance_taker")
    if taker:
        if taker.get("aggressive_buyers"):
            agg["taker_flow"] = "buying"
            bullish += 1
        elif taker.get("aggressive_sellers"):
            agg["taker_flow"] = "selling"
            bearish += 1
        total += 1

    agg["bullish_signals"] = bullish
    agg["bearish_signals"] = bearish
    agg["signal_count"] = total

    if total > 0:
        if bullish > bearish + 2:
            agg["overall_bias"] = "bullish"
        elif bearish > bullish + 2:
            agg["overall_bias"] = "bearish"
        else:
            agg["overall_bias"] = "neutral"

    return agg


# ─── COINBASE ORDER BOOK DEPTH ────────────────────────────────────

def fetch_coinbase_orderbook(pair, auth=None):
    """Fetch Coinbase order book depth for bid/ask imbalance analysis."""
    try:
        if auth:
            resp = auth.get(f"/api/v3/brokerage/product_book", params={"product_id": pair, "limit": 50})
        else:
            resp = requests.get(
                f"https://api.exchange.coinbase.com/products/{pair}/book",
                params={"level": 2},
                timeout=TIMEOUT,
            ).json()

        bids = resp.get("pricebook", {}).get("bids", resp.get("bids", []))
        asks = resp.get("pricebook", {}).get("asks", resp.get("asks", []))

        if not bids or not asks:
            return None

        # Calculate bid/ask depth imbalance (top 20 levels)
        bid_depth = sum(float(b.get("size", b[1]) if isinstance(b, dict) else b[1]) for b in bids[:20])
        ask_depth = sum(float(a.get("size", a[1]) if isinstance(a, dict) else a[1]) for a in asks[:20])

        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0

        best_bid = float(bids[0].get("price", bids[0][0]) if isinstance(bids[0], dict) else bids[0][0])
        best_ask = float(asks[0].get("price", asks[0][0]) if isinstance(asks[0], dict) else asks[0][0])
        spread = (best_ask - best_bid) / best_bid * 100

        return {
            "pair": pair,
            "bid_depth": round(bid_depth, 4),
            "ask_depth": round(ask_depth, 4),
            "imbalance": round(imbalance, 4),  # +1 = all bids, -1 = all asks
            "imbalance_signal": "buy_pressure" if imbalance > 0.15 else "sell_pressure" if imbalance < -0.15 else "balanced",
            "spread_pct": round(spread, 4),
            "best_bid": best_bid,
            "best_ask": best_ask,
        }
    except Exception as e:
        log.debug(f"Order book error {pair}: {e}")
        return None


# ─── REAL-TIME NEWS / SENTIMENT ──────────────────────────────────

def fetch_cryptopanic_news(api_key=None):
    """Fetch latest crypto news from CryptoPanic."""
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={
                "auth_token": api_key,
                "kind": "news",
                "filter": "hot",
                "currencies": "BTC,ETH,SOL,DOGE,AVAX,LINK",
            },
            timeout=TIMEOUT,
        )
        data = r.json()
        posts = data.get("results", [])[:10]
        news = []
        for p in posts:
            votes = p.get("votes", {})
            sentiment_score = (
                votes.get("positive", 0) * 1
                + votes.get("important", 0) * 0.5
                - votes.get("negative", 0) * 1
                - votes.get("toxic", 0) * 0.5
            )
            currencies = [c.get("code", "") for c in p.get("currencies", [])]
            news.append({
                "title": p.get("title", ""),
                "sentiment": sentiment_score,
                "currencies": currencies,
                "published": p.get("published_at", ""),
                "kind": p.get("kind", ""),
            })
        return news
    except Exception as e:
        log.debug(f"CryptoPanic error: {e}")
        return None
