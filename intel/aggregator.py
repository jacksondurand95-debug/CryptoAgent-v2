#!/usr/bin/env python3
"""Intel Aggregator — Merges all sub-agent signals into a single intel brief.

This is called by the main trading agent before making decisions.
It reads all intel/data/*.json files and produces a unified signal.

Can also be run standalone to generate intel/data/aggregate.json.
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, save_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("aggregator")

# Weight each intel source by its predictive value
SOURCE_WEIGHTS = {
    "liquidations": 3.0,   # Most actionable — cascade risk drives immediate price
    "whales": 2.5,         # Exchange flows predict medium-term direction
    "onchain": 2.0,        # Stablecoin flows predict macro direction
    "sentiment": 1.5,      # Contrarian — extreme fear = buy, extreme greed = sell
    "news": 1.0,           # Noise, but alpha events can be critical
}

# Maximum age in seconds before we consider data stale
MAX_AGE = {
    "liquidations": 1200,   # 20 min (runs every 15)
    "whales": 7200,         # 2 hours (runs every hour)
    "onchain": 28800,       # 8 hours (runs every 6h)
    "sentiment": 7200,      # 2 hours (runs every hour)
    "news": 14400,          # 4 hours (runs every 2h)
}


def load_intel():
    """Load all intel data files and check freshness."""
    sources = {}
    now = time.time()

    for name in SOURCE_WEIGHTS:
        filename = f"{name}.json"
        data = load_data(filename)

        if not data:
            log.warning(f"No data for {name}")
            continue

        # Check freshness
        meta = data.get("_meta", {})
        updated_ts = meta.get("updated_ts", 0)
        age = now - updated_ts
        max_age = MAX_AGE.get(name, 7200)

        if age > max_age:
            log.warning(f"{name} data is stale ({age/60:.0f} min old, max {max_age/60:.0f} min)")
            data["_stale"] = True
            data["_age_minutes"] = round(age / 60, 1)
        else:
            data["_stale"] = False
            data["_age_minutes"] = round(age / 60, 1)

        sources[name] = data

    return sources


def compute_aggregate(sources):
    """Compute the master aggregate signal from all intel sources.

    Returns a dict with:
    - score: -100 to +100
    - bias: bullish/bearish/neutral
    - strength: strong/moderate/weak
    - per_source: individual source signals
    - confidence: 0-1.0 based on data completeness
    """
    per_source = {}
    total_score = 0
    total_weight = 0
    available_sources = 0

    for name, weight in SOURCE_WEIGHTS.items():
        data = sources.get(name)
        if not data:
            per_source[name] = {"status": "missing", "score": 0, "bias": "neutral"}
            continue

        agg = data.get("aggregate", {})
        score = agg.get("score", 0)
        bias = agg.get("bias", "neutral")
        strength = agg.get("strength", "weak")

        # Reduce weight for stale data
        effective_weight = weight
        if data.get("_stale"):
            effective_weight *= 0.3
            per_source[name] = {
                "status": "stale",
                "score": score,
                "bias": bias,
                "strength": strength,
                "age_min": data.get("_age_minutes"),
                "effective_weight": round(effective_weight, 2),
            }
        else:
            per_source[name] = {
                "status": "fresh",
                "score": score,
                "bias": bias,
                "strength": strength,
                "age_min": data.get("_age_minutes"),
                "effective_weight": round(effective_weight, 2),
            }
            available_sources += 1

        total_score += score * effective_weight
        total_weight += effective_weight

    # Compute final
    if total_weight > 0:
        final_score = int(total_score / total_weight)
    else:
        final_score = 0

    final_score = max(-100, min(100, final_score))

    if final_score > 25:
        final_bias = "bullish"
    elif final_score < -25:
        final_bias = "bearish"
    else:
        final_bias = "neutral"

    final_strength = "strong" if abs(final_score) > 60 else "moderate" if abs(final_score) > 35 else "weak"

    # Confidence based on how many sources are available and fresh
    total_sources = len(SOURCE_WEIGHTS)
    confidence = available_sources / total_sources

    return {
        "score": final_score,
        "bias": final_bias,
        "strength": final_strength,
        "confidence": round(confidence, 2),
        "available_sources": available_sources,
        "total_sources": total_sources,
        "per_source": per_source,
    }


def extract_coin_signals(sources):
    """Extract per-coin signals from intel data."""
    coins = {"BTC": [], "ETH": [], "SOL": []}

    # Liquidation cascade risk per coin
    liqs = sources.get("liquidations", {})
    cascade = liqs.get("cascade_risk", {})
    for coin in coins:
        if coin in cascade:
            cr = cascade[coin]
            if cr.get("signal"):
                coins[coin].append(cr["signal"])

    # News per-coin sentiment
    news = sources.get("news", {})
    coin_sent = news.get("coin_sentiment", {})
    for coin in coins:
        if coin in coin_sent:
            cs = coin_sent[coin]
            score = cs.get("score", 0)
            if abs(score) > 15:
                coins[coin].append({
                    "bias": cs.get("bias", "neutral"),
                    "score": min(abs(score), 60),
                    "weight": 0.8,
                    "reason": f"News sentiment: {cs.get('articles', 0)} articles, score={score}",
                })

    # Whale data (BTC/ETH specific)
    whales = sources.get("whales", {})
    btc_w = whales.get("btc_whales", {})
    if btc_w and "signal" in btc_w:
        coins["BTC"].append(btc_w["signal"])

    eth_w = whales.get("eth_whales", {})
    if eth_w and "signal" in eth_w:
        coins["ETH"].append(eth_w["signal"])

    # Compute per-coin aggregate
    coin_results = {}
    for coin, signals in coins.items():
        if not signals:
            coin_results[coin] = {"score": 0, "bias": "neutral", "signals": 0}
            continue

        total = sum(
            s["score"] * s.get("weight", 1.0) *
            (1 if s["bias"] == "bullish" else -1 if s["bias"] == "bearish" else 0)
            for s in signals
        )
        weight = sum(s.get("weight", 1.0) for s in signals)
        score = int(total / weight) if weight > 0 else 0
        score = max(-100, min(100, score))

        coin_results[coin] = {
            "score": score,
            "bias": "bullish" if score > 20 else "bearish" if score < -20 else "neutral",
            "signals": len(signals),
            "details": [{"bias": s["bias"], "reason": s["reason"]} for s in signals],
        }

    return coin_results


def extract_alpha_events(sources):
    """Pull out critical alpha events that need immediate attention."""
    events = []

    # News alpha
    news = sources.get("news", {})
    for evt in news.get("alpha_events", [])[:5]:
        events.append({
            "type": "news",
            "title": evt.get("title", ""),
            "sentiment": evt.get("sentiment", "neutral"),
            "importance": evt.get("importance", "low"),
            "coins": evt.get("coins", []),
        })

    # Liquidation cascade warnings
    liqs = sources.get("liquidations", {})
    cascade = liqs.get("cascade_risk", {})
    for coin, cr in cascade.items():
        if cr.get("risk_level") in ("critical", "high"):
            events.append({
                "type": "liquidation_risk",
                "title": f"{coin} cascade risk: {cr.get('risk_level')} ({cr.get('risk_score')})",
                "sentiment": "bearish" if cr.get("signal", {}).get("bias") == "bearish" else "bullish",
                "importance": "critical" if cr.get("risk_level") == "critical" else "high",
                "coins": [coin],
            })

    return events


def get_intel_brief():
    """Public API: Returns complete intel brief for the trading agent.

    This is the function the main agent.py should call.
    """
    sources = load_intel()
    aggregate = compute_aggregate(sources)
    coin_signals = extract_coin_signals(sources)
    alpha = extract_alpha_events(sources)

    return {
        "aggregate": aggregate,
        "coins": coin_signals,
        "alpha_events": alpha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run():
    """Standalone execution — generates aggregate.json."""
    log.info("=" * 50)
    log.info("INTEL AGGREGATOR — Generating brief")
    log.info("=" * 50)

    brief = get_intel_brief()

    # Log summary
    agg = brief["aggregate"]
    log.info(f"MASTER SIGNAL: score={agg['score']} bias={agg['bias']} "
             f"strength={agg['strength']} confidence={agg['confidence']}")

    for source, info in agg.get("per_source", {}).items():
        log.info(f"  {source}: {info.get('status')} score={info.get('score')} "
                 f"bias={info.get('bias')} weight={info.get('effective_weight', 'N/A')}")

    for coin, info in brief.get("coins", {}).items():
        log.info(f"  {coin}: score={info['score']} bias={info['bias']} ({info['signals']} signals)")

    if brief.get("alpha_events"):
        log.info("ALPHA EVENTS:")
        for evt in brief["alpha_events"]:
            log.info(f"  [{evt['importance']}] {evt['title'][:80]}")

    save_data("aggregate.json", brief)
    log.info("Aggregator complete.")
    return brief


if __name__ == "__main__":
    run()
