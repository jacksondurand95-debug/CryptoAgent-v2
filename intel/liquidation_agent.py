#!/usr/bin/env python3
"""Intel Sub-Agent: Liquidation Monitor.

Sources (all free, no API keys required):
1. OKX — Liquidation orders, funding rates, OI (free public API)
2. Binance — Liquidation stream fallback via REST (free)
3. Bybit — Liquidation data (free public API)
4. Derived: Liquidation magnet estimation from price levels

Runs: Every 15 minutes via GitHub Actions
Output: intel/data/liquidations.json
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import retry_get, load_data, save_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("liquidation")

PAIRS = {
    "BTC": {"okx": "BTC-USDT-SWAP", "bybit": "BTCUSDT"},
    "ETH": {"okx": "ETH-USDT-SWAP", "bybit": "ETHUSDT"},
    "SOL": {"okx": "SOL-USDT-SWAP", "bybit": "SOLUSDT"},
}

# Leverage levels for liquidation magnet estimation
LEVERAGE_LEVELS = [3, 5, 10, 25, 50, 100]


def fetch_okx_funding_oi():
    """OKX funding rates + open interest for all tracked pairs.

    This runs every 15 min so we catch rapid OI changes that signal liquidations.
    """
    log.info("Fetching OKX funding + OI...")
    results = {}

    for coin, ids in PAIRS.items():
        inst_id = ids["okx"]
        pair_data = {}

        # Funding rate
        r = retry_get("https://www.okx.com/api/v5/public/funding-rate",
                       params={"instId": inst_id}, timeout=10)
        if r:
            data = r.json().get("data", [])
            if data:
                rate = float(data[0].get("fundingRate", 0) or 0)
                next_rate = float(data[0].get("nextFundingRate", 0) or 0)
                pair_data["funding"] = {
                    "current_pct": round(rate * 100, 4),
                    "next_pct": round(next_rate * 100, 4),
                    "extreme_negative": rate < -0.0001,
                    "extreme_positive": rate > 0.0003,
                }

        time.sleep(0.3)

        # Open interest
        r2 = retry_get("https://www.okx.com/api/v5/public/open-interest",
                        params={"instType": "SWAP", "instId": inst_id}, timeout=10)
        if r2:
            data = r2.json().get("data", [])
            if data:
                oi = float(data[0].get("oi", 0))
                oi_usd = float(data[0].get("oiUsd", 0))
                pair_data["open_interest"] = {
                    "contracts": oi,
                    "usd": oi_usd,
                    "usd_m": round(oi_usd / 1e6, 1),
                }

        time.sleep(0.3)

        # Funding rate history (last 10 periods)
        r3 = retry_get("https://www.okx.com/api/v5/public/funding-rate-history",
                        params={"instId": inst_id, "limit": 10}, timeout=10)
        if r3:
            hist = r3.json().get("data", [])
            rates = []
            for h in hist:
                realized = h.get("realizedRate")
                if realized:
                    rates.append(round(float(realized) * 100, 4))
            pair_data["funding_history"] = rates

            # Calculate streaks
            neg_streak = 0
            for rate in rates:
                if rate < 0:
                    neg_streak += 1
                else:
                    break
            pos_streak = 0
            for rate in rates:
                if rate > 0:
                    pos_streak += 1
                else:
                    break
            pair_data["neg_funding_streak"] = neg_streak
            pair_data["pos_funding_streak"] = pos_streak

        time.sleep(0.3)

        # Long/Short ratio
        r4 = retry_get(f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio/{coin}",
                        params={"period": "5m"}, timeout=10)
        if r4:
            ls_data = r4.json().get("data", [])
            if ls_data:
                current_ls = float(ls_data[0][1]) if len(ls_data[0]) > 1 else 1.0
                pair_data["long_short_ratio"] = round(current_ls, 3)
                pair_data["crowd_long"] = current_ls > 1.3
                pair_data["crowd_short"] = current_ls < 0.7

        time.sleep(0.3)

        # Taker buy/sell volume
        r5 = retry_get("https://www.okx.com/api/v5/rubik/stat/taker-volume",
                        params={"ccy": coin, "instType": "CONTRACTS", "period": "5m"},
                        timeout=10)
        if r5:
            tv_data = r5.json().get("data", [])
            if tv_data and len(tv_data) >= 6:
                # Recent 30 minutes (6 x 5min)
                total_buy = sum(float(d[1]) for d in tv_data[:6])
                total_sell = sum(float(d[2]) for d in tv_data[:6])
                ratio = total_buy / total_sell if total_sell > 0 else 1.0

                pair_data["taker_30m"] = {
                    "buy_vol": round(total_buy, 0),
                    "sell_vol": round(total_sell, 0),
                    "ratio": round(ratio, 3),
                    "aggressive_buyers": ratio > 1.15,
                    "aggressive_sellers": ratio < 0.85,
                }

        results[coin] = pair_data
        time.sleep(0.5)

    return results


def fetch_bybit_liquidations():
    """Bybit recent liquidations — free public API."""
    log.info("Fetching Bybit liquidations...")
    results = {}

    for coin, ids in PAIRS.items():
        symbol = ids["bybit"]

        # Bybit public liquidation endpoint
        r = retry_get("https://api.bybit.com/v5/market/recent-trade",
                       params={"category": "linear", "symbol": symbol, "limit": 50},
                       timeout=10)
        if not r:
            continue

        trades = r.json().get("result", {}).get("list", [])

        # Bybit doesn't have a direct liquidation endpoint in v5 public API
        # But we can detect large trades as potential liquidation-related
        large_trades = []
        for t in trades:
            qty = float(t.get("size", 0))
            price = float(t.get("price", 0))
            usd_value = qty * price
            side = t.get("side", "")

            if usd_value > 100_000:  # >$100k = notable
                large_trades.append({
                    "price": price,
                    "usd": round(usd_value, 0),
                    "side": side,
                    "time": t.get("time", ""),
                })

        results[coin] = {
            "large_trades": large_trades[:10],
            "large_buy_usd": sum(t["usd"] for t in large_trades if t["side"] == "Buy"),
            "large_sell_usd": sum(t["usd"] for t in large_trades if t["side"] == "Sell"),
        }

        time.sleep(0.5)

    return results


def estimate_liquidation_levels(okx_data):
    """Estimate where liquidation clusters are based on current market state.

    Uses funding rate direction + crowd positioning to estimate where stops are.
    """
    log.info("Estimating liquidation levels...")
    results = {}

    for coin, data in okx_data.items():
        # Get current price
        r = retry_get(f"https://www.okx.com/api/v5/market/ticker?instId={PAIRS[coin]['okx']}")
        if not r:
            continue

        ticker = r.json().get("data", [])
        if not ticker:
            continue

        price = float(ticker[0].get("last", 0))
        high_24h = float(ticker[0].get("high24h", price))
        low_24h = float(ticker[0].get("low24h", price))

        if not price:
            continue

        # Estimate liquidation levels for common leverage
        long_liqs = {}  # Price levels where longs get liquidated (below current)
        short_liqs = {}  # Price levels where shorts get liquidated (above current)

        for lev in LEVERAGE_LEVELS:
            # Simplified liquidation price estimation
            # Long liq ~ entry * (1 - 1/leverage) assuming entry near recent high
            long_liq = round(high_24h * (1 - 0.85 / lev), 2)
            if long_liq < price:
                long_liqs[f"{lev}x"] = long_liq

            # Short liq ~ entry * (1 + 1/leverage) assuming entry near recent low
            short_liq = round(low_24h * (1 + 0.85 / lev), 2)
            if short_liq > price:
                short_liqs[f"{lev}x"] = short_liq

        # Find nearest liquidation magnet
        nearest_long_liq = max(long_liqs.values()) if long_liqs else 0
        nearest_short_liq = min(short_liqs.values()) if short_liqs else float("inf")

        dist_to_long_liq = abs(price - nearest_long_liq) / price if nearest_long_liq else 1
        dist_to_short_liq = abs(nearest_short_liq - price) / price if nearest_short_liq < float("inf") else 1

        # Determine which liquidation cluster is more likely to be hit
        ls_ratio = data.get("long_short_ratio", 1.0)
        funding = data.get("funding", {}).get("current_pct", 0)

        if ls_ratio > 1.3 and funding > 0.02:
            # Crowd is long + paying funding = long liquidations more likely
            magnet_direction = "down"
            magnet_target = nearest_long_liq
            magnet_distance = dist_to_long_liq
        elif ls_ratio < 0.7 and funding < -0.01:
            # Crowd is short + shorts paying = short squeeze more likely
            magnet_direction = "up"
            magnet_target = nearest_short_liq if nearest_short_liq < float("inf") else 0
            magnet_distance = dist_to_short_liq
        else:
            magnet_direction = "none"
            magnet_target = 0
            magnet_distance = 1

        results[coin] = {
            "price": price,
            "high_24h": high_24h,
            "low_24h": low_24h,
            "long_liquidations": long_liqs,
            "short_liquidations": short_liqs,
            "nearest_long_liq": nearest_long_liq,
            "nearest_short_liq": nearest_short_liq if nearest_short_liq < float("inf") else None,
            "magnet_direction": magnet_direction,
            "magnet_target": magnet_target,
            "magnet_distance_pct": round(magnet_distance * 100, 2),
        }

        time.sleep(0.3)

    return results


def compute_cascade_risk(okx_data, liq_levels):
    """Estimate probability of a liquidation cascade.

    High risk when:
    - High OI + extreme crowd positioning
    - Funding very extreme
    - Price near liquidation cluster
    """
    log.info("Computing cascade risk...")
    results = {}

    for coin in PAIRS:
        risk_score = 0
        risk_factors = []

        okx = okx_data.get(coin, {})
        levels = liq_levels.get(coin, {})

        # Factor 1: Crowd positioning extremes
        ls = okx.get("long_short_ratio", 1.0)
        if ls > 1.5:
            risk_score += 25
            risk_factors.append(f"L/S={ls:.2f} (extreme long)")
        elif ls < 0.6:
            risk_score += 25
            risk_factors.append(f"L/S={ls:.2f} (extreme short)")
        elif ls > 1.2 or ls < 0.8:
            risk_score += 10

        # Factor 2: Funding rate extreme
        funding = okx.get("funding", {}).get("current_pct", 0)
        if abs(funding) > 0.05:
            risk_score += 30
            risk_factors.append(f"FR={funding:.4f}% (extreme)")
        elif abs(funding) > 0.02:
            risk_score += 15
            risk_factors.append(f"FR={funding:.4f}% (elevated)")

        # Factor 3: Price near liquidation cluster
        magnet_dist = levels.get("magnet_distance_pct", 100)
        if magnet_dist < 2:
            risk_score += 30
            risk_factors.append(f"Price {magnet_dist:.1f}% from liq cluster")
        elif magnet_dist < 5:
            risk_score += 15
            risk_factors.append(f"Price {magnet_dist:.1f}% from liq cluster")

        # Factor 4: Taker imbalance
        taker = okx.get("taker_30m", {})
        ratio = taker.get("ratio", 1.0)
        if ratio > 1.3 or ratio < 0.7:
            risk_score += 15
            risk_factors.append(f"Taker ratio={ratio:.2f} (imbalanced)")

        risk_score = min(risk_score, 100)

        if risk_score >= 70:
            risk_level = "critical"
        elif risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 30:
            risk_level = "moderate"
        else:
            risk_level = "low"

        results[coin] = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
        }

        # Signal: high cascade risk = incoming volatility
        if risk_score >= 60:
            magnet_dir = levels.get("magnet_direction", "none")
            if magnet_dir == "down":
                results[coin]["signal"] = {
                    "bias": "bearish", "score": risk_score, "weight": 2.5,
                    "reason": f"{coin} cascade risk {risk_level} ({risk_score}) — longs at risk"
                }
            elif magnet_dir == "up":
                results[coin]["signal"] = {
                    "bias": "bullish", "score": risk_score, "weight": 2.5,
                    "reason": f"{coin} cascade risk {risk_level} ({risk_score}) — short squeeze"
                }
            else:
                results[coin]["signal"] = {
                    "bias": "neutral", "score": risk_score // 2, "weight": 1.0,
                    "reason": f"{coin} cascade risk {risk_level} ({risk_score}) — direction unclear"
                }

    return results


def run():
    """Main execution."""
    log.info("=" * 50)
    log.info("LIQUIDATION AGENT — Starting collection")
    log.info("=" * 50)

    existing = load_data("liquidations.json")
    history = existing.get("history", [])

    okx_data = fetch_okx_funding_oi()
    bybit = fetch_bybit_liquidations()
    liq_levels = estimate_liquidation_levels(okx_data)
    cascade = compute_cascade_risk(okx_data, liq_levels)

    # Aggregate signals from all coins
    signals = []
    for coin in PAIRS:
        coin_cascade = cascade.get(coin, {})
        if "signal" in coin_cascade:
            signals.append(coin_cascade["signal"])

    if signals:
        total_score = sum(
            s["score"] * s["weight"] * (1 if s["bias"] == "bullish" else -1 if s["bias"] == "bearish" else 0)
            for s in signals
        )
        total_weight = sum(s["weight"] for s in signals)
        agg_score = int(total_score / total_weight) if total_weight > 0 else 0
        agg_score = max(-100, min(100, agg_score))
    else:
        agg_score = 0

    agg_bias = "bullish" if agg_score > 25 else "bearish" if agg_score < -25 else "neutral"
    agg_strength = "strong" if abs(agg_score) > 60 else "moderate" if abs(agg_score) > 35 else "weak"

    now = datetime.now(timezone.utc)
    snapshot = {
        "timestamp": now.isoformat(),
        "ts": int(time.time()),
        "score": agg_score,
        "bias": agg_bias,
        "cascade_risks": {coin: cascade.get(coin, {}).get("risk_score", 0) for coin in PAIRS},
    }
    history.append(snapshot)

    result = {
        "aggregate": {
            "score": agg_score,
            "bias": agg_bias,
            "strength": agg_strength,
            "signals": [{"bias": s["bias"], "score": s["score"], "reason": s["reason"]} for s in signals],
        },
        "okx": okx_data,
        "bybit": bybit,
        "liquidation_levels": liq_levels,
        "cascade_risk": cascade,
        "history": history,
    }

    save_data("liquidations.json", result)

    for coin in PAIRS:
        cr = cascade.get(coin, {})
        log.info(f"{coin}: cascade_risk={cr.get('risk_score', 0)} "
                 f"({cr.get('risk_level', 'unknown')}) "
                 f"factors={cr.get('risk_factors', [])}")

    log.info(f"AGGREGATE: score={agg_score} bias={agg_bias} strength={agg_strength}")
    log.info("Liquidation agent complete.")
    return result


if __name__ == "__main__":
    run()
