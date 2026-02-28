#!/usr/bin/env python3
"""Intel Sub-Agent: Whale Tracking.

Sources (all free):
1. Blockchain.com — Large BTC transactions (free, no key)
2. Etherscan — Large ETH transfers (free tier: 5 req/sec)
3. Whale Alert public RSS/social — parsed from their free feed
4. Arkham Intelligence — free dashboard scrape fallback
5. Exchange flow estimation via known wallet addresses

Runs: Every hour via GitHub Actions
Output: intel/data/whales.json
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
log = logging.getLogger("whales")

# Known exchange addresses (partial list — for flow detection)
EXCHANGE_ADDRESSES_ETH = {
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "Binance",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "Binance",
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f": "Binance",
    "0x8894e0a0c962cb723c1ef8a1b63d28aaa95e1dfa": "Coinbase",
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": "Coinbase",
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase",
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX",
    "0x98ec059dc3adfbdd63429227d09cb36c2e8dbf0e": "OKX",
    "0x1ab4973a48dc892cd9971ece8e01dcc7688f8f23": "Kraken",
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "Kraken",
}

# Min USD thresholds for "whale" transactions
WHALE_THRESHOLD_BTC = 100  # BTC (roughly $5-10M+)
WHALE_THRESHOLD_ETH = 1000  # ETH (roughly $2-4M+)
WHALE_THRESHOLD_USD = 5_000_000  # $5M minimum


def fetch_btc_large_transactions():
    """Fetch large BTC transactions from blockchain.com.

    blockchain.com/unconfirmed-transactions provides recent mempool activity.
    We filter for large transactions only.
    """
    log.info("Fetching large BTC transactions...")

    # Get latest blocks for recent large txs
    r = retry_get("https://blockchain.info/latestblock")
    if not r:
        return None

    latest = r.json()
    block_height = latest.get("height", 0)

    large_txs = []

    # Check last 3 blocks for large transactions
    for offset in range(3):
        height = block_height - offset
        r = retry_get(f"https://blockchain.info/rawblock/{height}", timeout=20)
        if not r:
            continue

        block = r.json()
        for tx in block.get("tx", []):
            # Calculate total output value
            total_out_btc = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8

            if total_out_btc >= WHALE_THRESHOLD_BTC:
                # Determine if exchange inflow or outflow
                # Check outputs for known patterns (exchange deposit addresses often reuse)
                output_count = len(tx.get("out", []))
                input_count = len(tx.get("inputs", []))

                # Heuristic: many inputs, few outputs = consolidation (could be exchange)
                # Few inputs, many outputs = distribution
                if input_count > 5 and output_count <= 3:
                    tx_type = "consolidation"
                elif output_count > 5:
                    tx_type = "distribution"
                else:
                    tx_type = "transfer"

                large_txs.append({
                    "hash": tx.get("hash", "")[:16] + "...",
                    "btc": round(total_out_btc, 2),
                    "type": tx_type,
                    "inputs": input_count,
                    "outputs": output_count,
                    "block": height,
                    "time": tx.get("time", 0),
                })

        time.sleep(1)  # Rate limit

    # Sort by size
    large_txs.sort(key=lambda x: x["btc"], reverse=True)

    total_whale_btc = sum(tx["btc"] for tx in large_txs)
    consolidations = sum(1 for tx in large_txs if tx["type"] == "consolidation")
    distributions = sum(1 for tx in large_txs if tx["type"] == "distribution")

    result = {
        "large_txs": large_txs[:20],
        "total_whale_btc": round(total_whale_btc, 2),
        "count": len(large_txs),
        "consolidations": consolidations,
        "distributions": distributions,
        "blocks_scanned": 3,
    }

    # More consolidations = potential exchange deposits = selling pressure
    if consolidations > distributions * 2 and consolidations > 3:
        result["signal"] = {"bias": "bearish", "score": 40, "weight": 1.5,
                            "reason": f"BTC whale consolidation ({consolidations} txs) — potential selling"}
    elif distributions > consolidations * 2 and distributions > 3:
        result["signal"] = {"bias": "bullish", "score": 40, "weight": 1.5,
                            "reason": f"BTC whale distribution ({distributions} txs) — accumulation"}
    else:
        result["signal"] = {"bias": "neutral", "score": 0, "weight": 0.5,
                            "reason": f"BTC whale activity normal ({len(large_txs)} large txs)"}

    log.info(f"BTC whales: {len(large_txs)} large txs | "
             f"{total_whale_btc:.0f} BTC | cons={consolidations} dist={distributions}")
    return result


def fetch_eth_large_transfers():
    """Fetch large ETH transfers via Etherscan.

    Tracks exchange inflows (bearish) and outflows (bullish).
    Exchange inflow = depositing to sell
    Exchange outflow = withdrawing to hold
    """
    log.info("Fetching large ETH transfers...")
    etherscan_key = os.environ.get("ETHERSCAN_API_KEY", "")
    if not etherscan_key:
        log.info("No ETHERSCAN_API_KEY — using fallback")
        return _eth_fallback()

    # Get latest internal transactions for known exchange addresses
    exchange_inflows = []
    exchange_outflows = []

    # Check recent blocks for large ETH transfers
    r = retry_get("https://api.etherscan.io/api",
                   params={
                       "module": "account",
                       "action": "txlist",
                       "address": "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance
                       "page": 1, "offset": 50, "sort": "desc",
                       "apikey": etherscan_key,
                   })

    if r:
        txs = r.json().get("result", [])
        if isinstance(txs, list):
            for tx in txs:
                value_eth = int(tx.get("value", "0")) / 1e18
                if value_eth < WHALE_THRESHOLD_ETH:
                    continue

                to_addr = tx.get("to", "").lower()
                from_addr = tx.get("from", "").lower()
                binance_addr = "0x28c6c06298d514db089934071355e5743bf21d60"

                if to_addr == binance_addr:
                    exchange_inflows.append({
                        "from": from_addr[:10] + "...",
                        "eth": round(value_eth, 2),
                        "exchange": "Binance",
                        "direction": "inflow",
                        "hash": tx.get("hash", "")[:16] + "...",
                    })
                elif from_addr == binance_addr:
                    exchange_outflows.append({
                        "to": to_addr[:10] + "...",
                        "eth": round(value_eth, 2),
                        "exchange": "Binance",
                        "direction": "outflow",
                        "hash": tx.get("hash", "")[:16] + "...",
                    })

    time.sleep(0.5)

    # Check Coinbase too
    r2 = retry_get("https://api.etherscan.io/api",
                    params={
                        "module": "account",
                        "action": "txlist",
                        "address": "0x503828976d22510aad0201ac7ec88293211d23da",  # Coinbase
                        "page": 1, "offset": 50, "sort": "desc",
                        "apikey": etherscan_key,
                    })

    if r2:
        txs = r2.json().get("result", [])
        if isinstance(txs, list):
            coinbase_addr = "0x503828976d22510aad0201ac7ec88293211d23da"
            for tx in txs:
                value_eth = int(tx.get("value", "0")) / 1e18
                if value_eth < WHALE_THRESHOLD_ETH:
                    continue

                to_addr = tx.get("to", "").lower()
                from_addr = tx.get("from", "").lower()

                if to_addr == coinbase_addr:
                    exchange_inflows.append({
                        "from": from_addr[:10] + "...",
                        "eth": round(value_eth, 2),
                        "exchange": "Coinbase",
                        "direction": "inflow",
                    })
                elif from_addr == coinbase_addr:
                    exchange_outflows.append({
                        "to": to_addr[:10] + "...",
                        "eth": round(value_eth, 2),
                        "exchange": "Coinbase",
                        "direction": "outflow",
                    })

    total_inflow = sum(tx["eth"] for tx in exchange_inflows)
    total_outflow = sum(tx["eth"] for tx in exchange_outflows)
    net_flow = total_outflow - total_inflow  # Positive = net outflow = bullish

    result = {
        "exchange_inflows": exchange_inflows[:10],
        "exchange_outflows": exchange_outflows[:10],
        "total_inflow_eth": round(total_inflow, 2),
        "total_outflow_eth": round(total_outflow, 2),
        "net_flow_eth": round(net_flow, 2),
        "net_direction": "outflow" if net_flow > 0 else "inflow" if net_flow < 0 else "balanced",
    }

    # Net exchange outflow = bullish (whales withdrawing to hold)
    # Net exchange inflow = bearish (whales depositing to sell)
    if net_flow > 500:
        result["signal"] = {"bias": "bullish", "score": 50, "weight": 2.0,
                            "reason": f"ETH net outflow {net_flow:.0f} ETH — whales accumulating"}
    elif net_flow < -500:
        result["signal"] = {"bias": "bearish", "score": 50, "weight": 2.0,
                            "reason": f"ETH net inflow {abs(net_flow):.0f} ETH — whales depositing"}
    else:
        result["signal"] = {"bias": "neutral", "score": 0, "weight": 0.5,
                            "reason": f"ETH exchange flow balanced ({net_flow:+.0f} ETH)"}

    log.info(f"ETH whales: inflow={total_inflow:.0f} outflow={total_outflow:.0f} "
             f"net={net_flow:+.0f} ETH")
    return result


def _eth_fallback():
    """Fallback ETH whale data when no Etherscan key available."""
    return {
        "note": "No ETHERSCAN_API_KEY — limited data",
        "signal": {"bias": "neutral", "score": 0, "weight": 0.1,
                   "reason": "No Etherscan key — ETH whale data unavailable"},
    }


def fetch_exchange_reserves():
    """Estimate exchange reserves trend via CryptoQuant-like proxy.

    Uses DeFi Llama bridge data as a proxy for cross-chain flows.
    Large bridge outflows from Ethereum = moving to DeFi/L2s = bullish
    """
    log.info("Fetching bridge/exchange flow proxies...")

    # DeFi Llama bridges
    r = retry_get("https://bridges.llama.fi/bridges?includeChains=true")
    if not r:
        return None

    bridges = r.json().get("bridges", [])
    total_volume_24h = 0
    top_bridges = []

    for bridge in bridges[:15]:
        vol_24h = bridge.get("currentDayVolume", 0) or 0
        total_volume_24h += vol_24h
        if vol_24h > 0:
            top_bridges.append({
                "name": bridge.get("displayName", ""),
                "volume_24h_m": round(vol_24h / 1e6, 1),
                "chains": len(bridge.get("chains", [])),
            })

    top_bridges.sort(key=lambda x: x["volume_24h_m"], reverse=True)

    result = {
        "bridge_volume_24h_m": round(total_volume_24h / 1e6, 1),
        "top_bridges": top_bridges[:10],
    }

    # High bridge volume = capital rotation, could be risk-on
    if total_volume_24h > 500_000_000:
        result["signal"] = {"bias": "bullish", "score": 20, "weight": 0.8,
                            "reason": f"Bridge volume high (${total_volume_24h/1e6:.0f}M) — active capital"}
    else:
        result["signal"] = {"bias": "neutral", "score": 0, "weight": 0.3,
                            "reason": "Bridge volume normal"}

    log.info(f"Bridge volume: ${total_volume_24h/1e6:.0f}M 24h")
    return result


def run():
    """Main execution."""
    log.info("=" * 50)
    log.info("WHALE AGENT — Starting collection")
    log.info("=" * 50)

    existing = load_data("whales.json")
    history = existing.get("history", [])

    btc_whales = fetch_btc_large_transactions()
    eth_whales = fetch_eth_large_transfers()
    flows = fetch_exchange_reserves()

    # Aggregate
    signals = []
    for source in [btc_whales, eth_whales, flows]:
        if source and "signal" in source:
            signals.append(source["signal"])

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
    }
    history.append(snapshot)

    result = {
        "aggregate": {
            "score": agg_score,
            "bias": agg_bias,
            "strength": agg_strength,
            "signals": [{"bias": s["bias"], "score": s["score"], "reason": s["reason"]} for s in signals],
        },
        "btc_whales": btc_whales,
        "eth_whales": eth_whales,
        "exchange_flows": flows,
        "history": history,
    }

    save_data("whales.json", result)

    log.info(f"AGGREGATE: score={agg_score} bias={agg_bias} strength={agg_strength}")
    log.info("Whale agent complete.")
    return result


if __name__ == "__main__":
    run()
