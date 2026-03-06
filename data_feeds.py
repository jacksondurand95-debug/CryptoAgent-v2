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


# ─── MACRO / ON-CHAIN / SENTIMENT FEEDS ─────────────────────────

def fetch_binance_liquidations():
    """Binance forced liquidations — shows where the pain is."""
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/allForceOrders",
            params={"limit": 50},
            timeout=TIMEOUT,
        )
        rows = r.json()
        if not rows or isinstance(rows, dict):
            return None
        total_long_liq = 0
        total_short_liq = 0
        btc_liqs = 0
        for liq in rows:
            qty_usd = float(liq.get("price", 0)) * float(liq.get("origQty", 0))
            side = liq.get("side", "").upper()
            if side == "SELL":  # long got liquidated
                total_long_liq += qty_usd
            elif side == "BUY":  # short got liquidated
                total_short_liq += qty_usd
            if "BTC" in liq.get("symbol", ""):
                btc_liqs += 1
        total = total_long_liq + total_short_liq
        return {
            "source": "binance_liquidations",
            "count": len(rows),
            "long_liq_usd": round(total_long_liq, 2),
            "short_liq_usd": round(total_short_liq, 2),
            "bias": "longs_rekt" if total_long_liq > total_short_liq * 1.5 else
                    "shorts_rekt" if total_short_liq > total_long_liq * 1.5 else "balanced",
            "btc_liq_count": btc_liqs,
            "total_usd": round(total, 2),
        }
    except Exception as e:
        log.debug(f"Binance liquidations error: {e}")
        return None


def fetch_mempool_data():
    """Bitcoin mempool — unconfirmed txs, fee rates, congestion."""
    try:
        mempool_r = requests.get("https://mempool.space/api/mempool", timeout=TIMEOUT)
        fees_r = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=TIMEOUT)
        mempool = mempool_r.json()
        fees = fees_r.json()
        tx_count = mempool.get("count", 0)
        vsize = mempool.get("vsize", 0)
        fastest = fees.get("fastestFee", 0)
        half_hour = fees.get("halfHourFee", 0)
        hour = fees.get("hourFee", 0)
        return {
            "source": "mempool_space",
            "unconfirmed_txs": tx_count,
            "mempool_vsize_mb": round(vsize / 1_000_000, 2) if vsize else 0,
            "fastest_fee_sat": fastest,
            "half_hour_fee_sat": half_hour,
            "hour_fee_sat": hour,
            "congested": tx_count > 50000 or fastest > 50,
            "fee_signal": "high_activity" if fastest > 30 else "normal" if fastest > 5 else "dead",
        }
    except Exception as e:
        log.debug(f"Mempool error: {e}")
        return None


def fetch_defillama_tvl():
    """DeFiLlama — total TVL across chains."""
    try:
        r = requests.get("https://api.llama.fi/v2/chains", timeout=TIMEOUT)
        chains = r.json()
        if not chains or not isinstance(chains, list):
            return None
        total_tvl = sum(float(c.get("tvl", 0)) for c in chains)
        top_chains = sorted(chains, key=lambda c: float(c.get("tvl", 0)), reverse=True)[:5]
        return {
            "source": "defillama_tvl",
            "total_tvl_b": round(total_tvl / 1e9, 2),
            "top_chains": [
                {"name": c.get("name", "?"), "tvl_b": round(float(c.get("tvl", 0)) / 1e9, 2)}
                for c in top_chains
            ],
        }
    except Exception as e:
        log.debug(f"DeFiLlama TVL error: {e}")
        return None


def fetch_defillama_stablecoin_flows():
    """DeFiLlama — stablecoin market cap by chain (inflows/outflows proxy)."""
    try:
        r = requests.get("https://stablecoins.llama.fi/stablecoinchains", timeout=TIMEOUT)
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        total_mcap = sum(float(c.get("totalCirculatingUSD", {}).get("peggedUSD", 0)) for c in data)
        top = sorted(data, key=lambda c: float(c.get("totalCirculatingUSD", {}).get("peggedUSD", 0)), reverse=True)[:5]
        return {
            "source": "defillama_stablecoins",
            "total_stablecoin_mcap_b": round(total_mcap / 1e9, 2),
            "top_chains": [
                {"name": c.get("name", "?"), "stables_b": round(float(c.get("totalCirculatingUSD", {}).get("peggedUSD", 0)) / 1e9, 2)}
                for c in top
            ],
        }
    except Exception as e:
        log.debug(f"DeFiLlama stablecoin error: {e}")
        return None


def fetch_defillama_dex_volume():
    """DeFiLlama — aggregated DEX volumes."""
    try:
        r = requests.get("https://api.llama.fi/overview/dexs", timeout=TIMEOUT)
        data = r.json()
        if not data:
            return None
        total_24h = float(data.get("total24h", 0))
        total_change = float(data.get("change_1d", 0))
        return {
            "source": "defillama_dex",
            "total_dex_volume_24h_b": round(total_24h / 1e9, 2),
            "volume_change_1d_pct": round(total_change, 2),
            "high_volume": total_24h > 5e9,
        }
    except Exception as e:
        log.debug(f"DeFiLlama DEX error: {e}")
        return None


def fetch_btc_hashrate():
    """Bitcoin hash rate — network health indicator."""
    try:
        r = requests.get(
            "https://api.blockchain.info/charts/hash-rate",
            params={"timespan": "30days", "format": "json"},
            timeout=TIMEOUT,
        )
        data = r.json()
        values = data.get("values", [])
        if not values:
            return None
        current = values[-1].get("y", 0)
        week_ago = values[-7].get("y", 0) if len(values) >= 7 else current
        month_ago = values[0].get("y", 0)
        return {
            "source": "blockchain_info",
            "hashrate_eh": round(current / 1e6, 2),  # convert to EH/s
            "change_7d_pct": round(((current - week_ago) / week_ago) * 100, 2) if week_ago else 0,
            "change_30d_pct": round(((current - month_ago) / month_ago) * 100, 2) if month_ago else 0,
            "healthy": current > week_ago * 0.95,  # not dropping more than 5%
        }
    except Exception as e:
        log.debug(f"BTC hashrate error: {e}")
        return None


def fetch_eth_gas():
    """ETH gas prices — high gas = high on-chain activity."""
    try:
        r = requests.get(
            "https://api.etherscan.io/api",
            params={"module": "gastracker", "action": "gasoracle"},
            timeout=TIMEOUT,
        )
        data = r.json()
        result = data.get("result", {})
        if not result or isinstance(result, str):
            return None
        safe = float(result.get("SafeGasPrice", 0))
        propose = float(result.get("ProposeGasPrice", 0))
        fast = float(result.get("FastGasPrice", 0))
        return {
            "source": "etherscan_gas",
            "safe_gwei": safe,
            "propose_gwei": propose,
            "fast_gwei": fast,
            "high_activity": fast > 50,
            "signal": "bullish_activity" if fast > 40 else "normal" if fast > 10 else "low_activity",
        }
    except Exception as e:
        log.debug(f"ETH gas error: {e}")
        return None


def fetch_gbtc_premium():
    """GBTC premium/discount proxy via CoinGecko — ETF flow signal."""
    try:
        # Get BTC spot price
        btc_r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            timeout=TIMEOUT,
        )
        btc_price = btc_r.json().get("bitcoin", {}).get("usd", 0)
        if not btc_price:
            return None
        # Get GBTC price (traded as wrapped token / tracked by CoinGecko)
        gbtc_r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "grayscale-bitcoin-trust", "vs_currencies": "usd"},
            timeout=TIMEOUT,
        )
        gbtc_price = gbtc_r.json().get("grayscale-bitcoin-trust", {}).get("usd", 0)
        if not gbtc_price:
            return None
        # GBTC holds ~0.00089 BTC per share (approximate)
        nav_approx = btc_price * 0.00089
        premium_pct = ((gbtc_price - nav_approx) / nav_approx) * 100 if nav_approx > 0 else 0
        return {
            "source": "coingecko_gbtc",
            "btc_price": btc_price,
            "gbtc_price": gbtc_price,
            "nav_approx": round(nav_approx, 2),
            "premium_pct": round(premium_pct, 2),
            "signal": "strong_demand" if premium_pct > 2 else "discount" if premium_pct < -2 else "at_nav",
        }
    except Exception as e:
        log.debug(f"GBTC premium error: {e}")
        return None


def fetch_coinglass_fear_greed():
    """Coinglass Fear & Greed index — backup sentiment source."""
    try:
        r = requests.get(
            "https://api.coinglass.com/api/index/fear-greed-history",
            timeout=TIMEOUT,
        )
        data = r.json()
        rows = data.get("data", [])
        if not rows:
            return None
        latest = rows[-1] if isinstance(rows, list) else None
        if not latest:
            return None
        value = int(latest.get("value", 50))
        return {
            "source": "coinglass_fgi",
            "value": value,
            "classification": "extreme_fear" if value <= 20 else "fear" if value <= 40 else
                              "neutral" if value <= 60 else "greed" if value <= 80 else "extreme_greed",
            "contrarian_signal": "bullish" if value <= 25 else "bearish" if value >= 75 else "neutral",
        }
    except Exception as e:
        log.debug(f"Coinglass FGI error: {e}")
        return None


# ─── UNDERGROUND ALPHA FEEDS ─────────────────────────────────────

def fetch_dexscreener_trending():
    """DexScreener — trending tokens on DEX (smart money moving before CEX)."""
    try:
        r = requests.get(
            "https://api.dexscreener.com/token-boosts/latest/v1",
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        # Focus on tokens that are also on Coinbase
        cb_symbols = {"BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "PEPE", "SHIB",
                       "SUI", "NEAR", "RENDER", "FET", "INJ", "TIA", "SEI", "WIF"}
        hot = []
        for token in data[:30]:
            symbol = token.get("tokenAddress", "")
            desc = token.get("description", "")
            chain = token.get("chainId", "")
            amount = token.get("amount", 0)
            if amount > 100:
                hot.append({"chain": chain, "amount": amount, "desc": desc[:60]})
        return {
            "source": "dexscreener_trending",
            "boosted_tokens": len(data) if isinstance(data, list) else 0,
            "high_activity": hot[:5],
            "signal": "high_dex_activity" if len(hot) > 3 else "normal",
        }
    except Exception as e:
        log.debug(f"DexScreener error: {e}")
        return None


def fetch_coingecko_trending():
    """CoinGecko trending coins — what retail is piling into."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=TIMEOUT,
        )
        data = r.json()
        coins = data.get("coins", [])
        trending = []
        for c in coins[:10]:
            item = c.get("item", {})
            trending.append({
                "name": item.get("name", ""),
                "symbol": item.get("symbol", ""),
                "market_cap_rank": item.get("market_cap_rank"),
                "price_btc": item.get("price_btc", 0),
            })
        return {
            "source": "coingecko_trending",
            "trending_coins": trending,
            "count": len(trending),
        }
    except Exception as e:
        log.debug(f"CoinGecko trending error: {e}")
        return None


def fetch_binance_long_short_global():
    """Binance global long/short ratio — crowd positioning."""
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 5},
            timeout=TIMEOUT,
        )
        data = r.json()
        if not data or isinstance(data, dict):
            return None
        latest = data[0]
        ratio = float(latest.get("longShortRatio", 1.0))
        long_pct = float(latest.get("longAccount", 50))
        short_pct = float(latest.get("shortAccount", 50))
        # Check trend
        prev_ratio = float(data[-1].get("longShortRatio", 1.0)) if len(data) > 1 else ratio
        return {
            "source": "binance_global_ls",
            "long_short_ratio": round(ratio, 3),
            "long_pct": round(long_pct, 1),
            "short_pct": round(short_pct, 1),
            "trend": "longs_increasing" if ratio > prev_ratio else "shorts_increasing",
            "crowded_longs": long_pct > 65,
            "crowded_shorts": short_pct > 55,
            "contrarian_signal": "bearish" if long_pct > 65 else "bullish" if short_pct > 55 else "neutral",
        }
    except Exception as e:
        log.debug(f"Binance global L/S error: {e}")
        return None


def fetch_binance_oi_change():
    """Binance open interest changes — big OI spike = incoming volatility."""
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
        results = {}
        for sym in symbols:
            r = requests.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": sym, "period": "1h", "limit": 5},
                timeout=TIMEOUT,
            )
            data = r.json()
            if not data or isinstance(data, dict):
                continue
            latest_oi = float(data[0].get("sumOpenInterestValue", 0))
            prev_oi = float(data[-1].get("sumOpenInterestValue", 0)) if len(data) > 1 else latest_oi
            change_pct = ((latest_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
            results[sym] = {
                "oi_usd": round(latest_oi, 0),
                "change_pct": round(change_pct, 2),
                "surging": change_pct > 5,
                "dumping": change_pct < -5,
            }
        if not results:
            return None
        return {
            "source": "binance_oi_change",
            "symbols": results,
            "any_surging": any(v["surging"] for v in results.values()),
            "any_dumping": any(v["dumping"] for v in results.values()),
            "signal": "volatility_incoming" if any(v["surging"] or v["dumping"] for v in results.values()) else "stable",
        }
    except Exception as e:
        log.debug(f"Binance OI change error: {e}")
        return None


def fetch_whale_transactions():
    """Blockchain.com large BTC transactions — whale movement detection."""
    try:
        r = requests.get(
            "https://blockchain.info/unconfirmed-transactions?format=json",
            timeout=TIMEOUT,
        )
        data = r.json()
        txs = data.get("txs", [])
        large_txs = []
        for tx in txs:
            total_output = sum(out.get("value", 0) for out in tx.get("out", [])) / 1e8  # satoshi to BTC
            if total_output > 10:  # > 10 BTC
                large_txs.append({
                    "btc_amount": round(total_output, 2),
                    "outputs": len(tx.get("out", [])),
                    "hash": tx.get("hash", "")[:16],
                })
        large_txs.sort(key=lambda x: x["btc_amount"], reverse=True)
        whale_volume = sum(t["btc_amount"] for t in large_txs)
        return {
            "source": "blockchain_whale_txs",
            "large_tx_count": len(large_txs),
            "whale_btc_volume": round(whale_volume, 2),
            "top_txs": large_txs[:5],
            "whale_active": len(large_txs) > 5,
            "signal": "heavy_whale_movement" if whale_volume > 500 else
                      "moderate_whale_activity" if whale_volume > 100 else "quiet",
        }
    except Exception as e:
        log.debug(f"Whale TX error: {e}")
        return None


def fetch_options_data():
    """Deribit options data via public API — max pain and put/call ratio."""
    try:
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
            params={"currency": "BTC", "kind": "option"},
            timeout=TIMEOUT,
        )
        data = r.json()
        options = data.get("result", [])
        if not options:
            return None
        total_calls_oi = 0
        total_puts_oi = 0
        total_calls_vol = 0
        total_puts_vol = 0
        for opt in options:
            name = opt.get("instrument_name", "")
            oi = float(opt.get("open_interest", 0))
            vol = float(opt.get("volume", 0))
            if "-C" in name:
                total_calls_oi += oi
                total_calls_vol += vol
            elif "-P" in name:
                total_puts_oi += oi
                total_puts_vol += vol
        pc_ratio_oi = total_puts_oi / total_calls_oi if total_calls_oi > 0 else 1.0
        pc_ratio_vol = total_puts_vol / total_calls_vol if total_calls_vol > 0 else 1.0
        return {
            "source": "deribit_options",
            "put_call_ratio_oi": round(pc_ratio_oi, 3),
            "put_call_ratio_vol": round(pc_ratio_vol, 3),
            "total_calls_oi": round(total_calls_oi, 2),
            "total_puts_oi": round(total_puts_oi, 2),
            "sentiment": "bearish_hedging" if pc_ratio_oi > 1.2 else
                         "bullish_positioning" if pc_ratio_oi < 0.7 else "neutral",
            "signal": "extreme_puts" if pc_ratio_oi > 1.5 else
                      "extreme_calls" if pc_ratio_oi < 0.5 else "balanced",
        }
    except Exception as e:
        log.debug(f"Deribit options error: {e}")
        return None


def fetch_cmc_greed_index():
    """CoinMarketCap-style aggregated sentiment from multiple sources."""
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=3&format=json",
            timeout=TIMEOUT,
        )
        data = r.json()
        entries = data.get("data", [])
        if not entries:
            return None
        current = entries[0]
        value = int(current.get("value", 50))
        prev = int(entries[1].get("value", 50)) if len(entries) > 1 else value
        return {
            "source": "alternative_me_fng",
            "value": value,
            "previous": prev,
            "trend": "improving" if value > prev else "deteriorating" if value < prev else "stable",
            "classification": current.get("value_classification", ""),
            "contrarian_buy": value <= 20,
            "contrarian_sell": value >= 80,
        }
    except Exception as e:
        log.debug(f"Alt.me FNG error: {e}")
        return None


def fetch_macro_intel():
    """Fetch all macro/on-chain/sentiment data in parallel.

    Returns a combined dict with all available macro intelligence.
    Non-blocking — any feed that fails is silently skipped.
    """
    feeds = {
        "liquidations": fetch_binance_liquidations,
        "mempool": fetch_mempool_data,
        "tvl": fetch_defillama_tvl,
        "stablecoin_flows": fetch_defillama_stablecoin_flows,
        "dex_volume": fetch_defillama_dex_volume,
        "btc_hashrate": fetch_btc_hashrate,
        "eth_gas": fetch_eth_gas,
        "gbtc_premium": fetch_gbtc_premium,
        "coinglass_fgi": fetch_coinglass_fear_greed,
        # Underground alpha feeds
        "dex_trending": fetch_dexscreener_trending,
        "trending_coins": fetch_coingecko_trending,
        "global_long_short": fetch_binance_long_short_global,
        "oi_changes": fetch_binance_oi_change,
        "whale_txs": fetch_whale_transactions,
        "options": fetch_options_data,
        "alt_fng": fetch_cmc_greed_index,
    }

    results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=10) as pool:
        for key, fn in feeds.items():
            futures[key] = pool.submit(fn)

        for key, future in futures.items():
            try:
                result = future.result(timeout=TIMEOUT + 2)
                if result:
                    results[key] = result
            except Exception as e:
                log.debug(f"Macro feed {key} failed: {e}")

    results["feed_count"] = len(results)
    log.info(f"MACRO INTEL: {len(results) - 1}/{len(feeds)} feeds loaded")
    return results
