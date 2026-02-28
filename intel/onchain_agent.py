#!/usr/bin/env python3
"""Intel Sub-Agent: On-Chain Analytics.

Sources (all free, no API keys required):
1. DeFi Llama — TVL, stablecoin flows, DEX volume (completely free, no key)
2. Blockchain.com — Bitcoin on-chain metrics (free API)
3. Etherscan — ETH supply, gas (free tier: 5 req/sec, needs API key)
4. Mempool.space — Bitcoin mempool, fees (completely free)
5. DeFi Llama yields — yield farming rates (free)

Runs: Every 6 hours via GitHub Actions
Output: intel/data/onchain.json
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
log = logging.getLogger("onchain")


def fetch_stablecoin_flows():
    """DeFi Llama stablecoin market cap and flows — THE macro indicator.

    Rising stablecoin mcap = money flowing INTO crypto = bullish
    Falling = money exiting = bearish
    """
    log.info("Fetching stablecoin flows from DeFi Llama...")

    # Get all stablecoins
    r = retry_get("https://stablecoins.llama.fi/stablecoins?includePrices=false")
    if not r:
        return None

    data = r.json()
    stables = data.get("peggedAssets", [])

    target_names = {"Tether", "USD Coin", "Dai", "First Digital USD", "USDS", "USDE"}
    total_mcap = 0
    breakdown = {}

    for s in stables:
        name = s.get("name", "")
        if name in target_names:
            asset_mcap = 0
            chains = s.get("chainCirculating", {})
            for chain, chain_data in chains.items():
                current = chain_data.get("current", {})
                asset_mcap += current.get("peggedUSD", 0)
            total_mcap += asset_mcap
            if asset_mcap > 0:
                breakdown[name] = round(asset_mcap / 1e9, 2)

    # Get USDT history for trend analysis
    r2 = retry_get("https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1")
    week_change = 0
    month_change = 0
    if r2:
        history = r2.json()
        if history and len(history) >= 30:
            recent = history[-30:]
            mcaps = []
            for d in recent:
                total = sum(v.get("peggedUSD", 0) for v in d.get("totalCirculating", {}).values())
                mcaps.append(total)

            if len(mcaps) >= 7:
                current_val = mcaps[-1]
                week_ago = mcaps[-7]
                month_ago = mcaps[0]
                week_change = ((current_val - week_ago) / week_ago * 100) if week_ago else 0
                month_change = ((current_val - month_ago) / month_ago * 100) if month_ago else 0

    result = {
        "total_mcap_b": round(total_mcap / 1e9, 2),
        "breakdown": breakdown,
        "week_change_pct": round(week_change, 3),
        "month_change_pct": round(month_change, 3),
        "trend": "inflow" if week_change > 0.5 else "outflow" if week_change < -0.5 else "flat",
    }

    # Signal
    if week_change > 1.0:
        result["signal"] = {"bias": "bullish", "score": 70, "weight": 2.5,
                            "reason": f"Stablecoin inflow +{week_change:.2f}%/week"}
    elif week_change > 0.3:
        result["signal"] = {"bias": "bullish", "score": 35, "weight": 1.5,
                            "reason": f"Mild stablecoin inflow +{week_change:.2f}%/week"}
    elif week_change < -1.0:
        result["signal"] = {"bias": "bearish", "score": 70, "weight": 2.5,
                            "reason": f"Stablecoin outflow {week_change:.2f}%/week"}
    elif week_change < -0.3:
        result["signal"] = {"bias": "bearish", "score": 35, "weight": 1.5,
                            "reason": f"Mild stablecoin outflow {week_change:.2f}%/week"}
    else:
        result["signal"] = {"bias": "neutral", "score": 0, "weight": 0.5,
                            "reason": "Stablecoin flows flat"}

    log.info(f"Stablecoins: ${total_mcap/1e9:.1f}B | "
             f"7d: {week_change:+.2f}% | 30d: {month_change:+.2f}%")
    return result


def fetch_defi_tvl():
    """DeFi Llama total TVL — measures overall DeFi health."""
    log.info("Fetching DeFi TVL from DeFi Llama...")

    r = retry_get("https://api.llama.fi/v2/historicalChainTvl")
    if not r:
        return None

    data = r.json()
    if not data or len(data) < 30:
        return None

    recent = data[-30:]
    tvls = [d.get("tvl", 0) for d in recent]
    current = tvls[-1] if tvls else 0
    week_ago = tvls[-7] if len(tvls) >= 7 else tvls[0]
    month_ago = tvls[0]

    week_change = ((current - week_ago) / week_ago * 100) if week_ago else 0
    month_change = ((current - month_ago) / month_ago * 100) if month_ago else 0

    # Get chain breakdown
    r2 = retry_get("https://api.llama.fi/v2/chains")
    chains = {}
    if r2:
        for chain in r2.json()[:10]:
            chains[chain.get("name", "")] = round(chain.get("tvl", 0) / 1e9, 2)

    result = {
        "total_tvl_b": round(current / 1e9, 2),
        "week_change_pct": round(week_change, 2),
        "month_change_pct": round(month_change, 2),
        "top_chains": chains,
        "trend": "growing" if week_change > 2 else "shrinking" if week_change < -2 else "stable",
    }

    if week_change > 5:
        result["signal"] = {"bias": "bullish", "score": 50, "weight": 1.5,
                            "reason": f"TVL surging +{week_change:.1f}%/week"}
    elif week_change < -5:
        result["signal"] = {"bias": "bearish", "score": 50, "weight": 1.5,
                            "reason": f"TVL declining {week_change:.1f}%/week"}
    else:
        result["signal"] = {"bias": "neutral", "score": 0, "weight": 0.5,
                            "reason": f"TVL stable ({week_change:+.1f}%)"}

    log.info(f"DeFi TVL: ${current/1e9:.1f}B | 7d: {week_change:+.1f}% | 30d: {month_change:+.1f}%")
    return result


def fetch_dex_volume():
    """DeFi Llama DEX volume — trading activity indicator."""
    log.info("Fetching DEX volume from DeFi Llama...")

    r = retry_get("https://api.llama.fi/overview/dexs?excludeTotalDataChart=false"
                   "&excludeTotalDataChartBreakdown=true&dataType=dailyVolume")
    if not r:
        return None

    data = r.json()
    total_24h = data.get("total24h", 0)
    total_7d = data.get("total7d", 0)
    change_1d = data.get("change_1d", 0)
    change_7d = data.get("change_7d", 0)

    # Top protocols by volume
    protocols = []
    for p in data.get("protocols", [])[:10]:
        protocols.append({
            "name": p.get("name", ""),
            "volume_24h": round(p.get("total24h", 0) / 1e6, 1),
            "change_1d": round(p.get("change_1d", 0) or 0, 1),
        })

    result = {
        "total_24h_m": round(total_24h / 1e6, 1),
        "total_7d_m": round(total_7d / 1e6, 1),
        "change_1d_pct": round(change_1d or 0, 1),
        "change_7d_pct": round(change_7d or 0, 1),
        "top_protocols": protocols,
    }

    # High volume = high activity = increased volatility opportunity
    if change_1d and change_1d > 30:
        result["signal"] = {"bias": "neutral", "score": 30, "weight": 1.0,
                            "reason": f"DEX volume spike +{change_1d:.0f}% — increased volatility"}
    else:
        result["signal"] = {"bias": "neutral", "score": 0, "weight": 0.3,
                            "reason": "DEX volume normal"}

    log.info(f"DEX volume: ${total_24h/1e6:.0f}M 24h | "
             f"1d: {change_1d:+.1f}% | 7d: {change_7d:+.1f}%")
    return result


def fetch_btc_onchain():
    """Bitcoin on-chain metrics from blockchain.com and mempool.space."""
    log.info("Fetching BTC on-chain data...")
    result = {}

    # Mempool.space — fees and mempool (completely free, no auth)
    r = retry_get("https://mempool.space/api/v1/fees/recommended")
    if r:
        fees = r.json()
        result["btc_fees"] = {
            "fastest": fees.get("fastestFee", 0),
            "half_hour": fees.get("halfHourFee", 0),
            "hour": fees.get("hourFee", 0),
            "economy": fees.get("economyFee", 0),
            "minimum": fees.get("minimumFee", 0),
        }
        # High fees = high network activity = potential top
        fastest = fees.get("fastestFee", 0)
        if fastest > 100:
            result["fee_signal"] = {"bias": "bearish", "score": 30, "weight": 0.8,
                                    "reason": f"BTC fees very high ({fastest} sat/vB) — peak activity"}
        elif fastest < 5:
            result["fee_signal"] = {"bias": "neutral", "score": 0, "weight": 0.3,
                                    "reason": f"BTC fees low ({fastest} sat/vB) — quiet network"}

    # Mempool size
    r2 = retry_get("https://mempool.space/api/mempool")
    if r2:
        mempool = r2.json()
        result["btc_mempool"] = {
            "count": mempool.get("count", 0),
            "vsize": mempool.get("vsize", 0),
            "total_fee": mempool.get("total_fee", 0),
        }

    # Blockchain.com — hash rate and difficulty
    r3 = retry_get("https://api.blockchain.info/q/hashrate")
    if r3:
        try:
            hashrate = float(r3.text)
            result["btc_hashrate"] = hashrate
        except ValueError:
            pass

    time.sleep(1)

    # Bitcoin blocks
    r4 = retry_get("https://mempool.space/api/v1/mining/hashrate/1w")
    if r4:
        data = r4.json()
        if data.get("hashrates"):
            recent = data["hashrates"][-7:]
            hashrates = [h.get("avgHashrate", 0) for h in recent]
            if hashrates:
                current_hr = hashrates[-1]
                avg_hr = sum(hashrates) / len(hashrates)
                result["hashrate_trend"] = {
                    "current_eh": round(current_hr / 1e18, 1),
                    "avg_7d_eh": round(avg_hr / 1e18, 1),
                    "rising": current_hr > avg_hr,
                }

    log.info(f"BTC on-chain: fees={result.get('btc_fees', {}).get('fastest', '?')} sat/vB | "
             f"mempool={result.get('btc_mempool', {}).get('count', '?')} tx")
    return result


def fetch_eth_onchain():
    """Ethereum on-chain metrics."""
    log.info("Fetching ETH on-chain data...")
    result = {}

    etherscan_key = os.environ.get("ETHERSCAN_API_KEY", "")

    # ETH gas from Etherscan free tier (5 req/sec)
    if etherscan_key:
        r = retry_get("https://api.etherscan.io/api",
                       params={"module": "gastracker", "action": "gasoracle",
                               "apikey": etherscan_key})
        if r:
            gas_data = r.json().get("result", {})
            result["eth_gas"] = {
                "safe": int(gas_data.get("SafeGasPrice", 0)),
                "propose": int(gas_data.get("ProposeGasPrice", 0)),
                "fast": int(gas_data.get("FastGasPrice", 0)),
            }
            fast = int(gas_data.get("FastGasPrice", 0))
            if fast > 100:
                result["gas_signal"] = {"bias": "neutral", "score": 20, "weight": 0.5,
                                        "reason": f"ETH gas high ({fast} gwei) — congested"}

        time.sleep(0.5)

        # ETH supply
        r2 = retry_get("https://api.etherscan.io/api",
                        params={"module": "stats", "action": "ethsupply2",
                                "apikey": etherscan_key})
        if r2:
            supply = r2.json().get("result", {})
            result["eth_supply"] = {
                "total_eth": round(float(supply.get("EthSupply", 0)) / 1e18, 0),
                "staking_eth": round(float(supply.get("Eth2Staking", 0)) / 1e18, 0),
                "burnt_eth": round(float(supply.get("BurntFees", 0)) / 1e18, 0),
            }
    else:
        log.info("No ETHERSCAN_API_KEY — skipping Etherscan calls")

    # Beaconcha.in — ETH staking stats (free)
    r3 = retry_get("https://beaconcha.in/api/v1/epoch/latest")
    if r3:
        epoch = r3.json().get("data", {})
        if epoch:
            result["eth_staking"] = {
                "active_validators": epoch.get("validatorscount", 0),
                "participation_rate": round(epoch.get("globalparticipationrate", 0) * 100, 2),
            }

    log.info(f"ETH on-chain: gas={result.get('eth_gas', {}).get('fast', '?')} gwei | "
             f"validators={result.get('eth_staking', {}).get('active_validators', '?')}")
    return result


def fetch_yield_data():
    """DeFi Llama yields — DeFi yield rates indicate risk appetite."""
    log.info("Fetching DeFi yields from DeFi Llama...")

    r = retry_get("https://yields.llama.fi/pools")
    if not r:
        return None

    pools = r.json().get("data", [])

    # Filter for major stablecoin yields (USDC, USDT, DAI on major chains)
    stable_yields = []
    for pool in pools:
        symbol = pool.get("symbol", "").upper()
        chain = pool.get("chain", "")
        tvl = pool.get("tvlUsd", 0)
        apy = pool.get("apy", 0)

        if tvl < 10_000_000:  # Only pools > $10M TVL
            continue
        if not any(s in symbol for s in ["USDC", "USDT", "DAI"]):
            continue
        if apy is None or apy <= 0 or apy > 100:  # Filter unrealistic
            continue

        stable_yields.append({
            "pool": pool.get("pool", ""),
            "project": pool.get("project", ""),
            "chain": chain,
            "symbol": symbol,
            "tvl_m": round(tvl / 1e6, 1),
            "apy": round(apy, 2),
        })

    if not stable_yields:
        return None

    # Sort by TVL and get top 20
    stable_yields.sort(key=lambda x: x["tvl_m"], reverse=True)
    top_pools = stable_yields[:20]

    apys = [p["apy"] for p in top_pools]
    avg_apy = sum(apys) / len(apys)
    median_apy = sorted(apys)[len(apys) // 2]

    result = {
        "avg_stable_apy": round(avg_apy, 2),
        "median_stable_apy": round(median_apy, 2),
        "top_pools": top_pools[:10],
        "pool_count": len(stable_yields),
    }

    # High yields = high demand for leverage = potential overheating
    if avg_apy > 15:
        result["signal"] = {"bias": "bearish", "score": 30, "weight": 0.8,
                            "reason": f"DeFi yields elevated ({avg_apy:.1f}%) — overleveraged market"}
    elif avg_apy < 3:
        result["signal"] = {"bias": "bullish", "score": 20, "weight": 0.5,
                            "reason": f"DeFi yields depressed ({avg_apy:.1f}%) — capitulation"}
    else:
        result["signal"] = {"bias": "neutral", "score": 0, "weight": 0.3,
                            "reason": f"DeFi yields normal ({avg_apy:.1f}%)"}

    log.info(f"DeFi yields: avg={avg_apy:.1f}% median={median_apy:.1f}% "
             f"({len(stable_yields)} pools)")
    return result


def run():
    """Main execution — gather all on-chain data."""
    log.info("=" * 50)
    log.info("ON-CHAIN AGENT — Starting collection")
    log.info("=" * 50)

    existing = load_data("onchain.json")
    history = existing.get("history", [])

    stablecoins = fetch_stablecoin_flows()
    tvl = fetch_defi_tvl()
    dex = fetch_dex_volume()
    btc = fetch_btc_onchain()
    eth = fetch_eth_onchain()
    yields = fetch_yield_data()

    # Aggregate signals
    signals = []
    for source in [stablecoins, tvl, dex, btc, eth, yields]:
        if source and isinstance(source, dict):
            for key in ["signal", "fee_signal", "gas_signal"]:
                if key in source:
                    signals.append(source[key])

    # Compute aggregate
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
        "stablecoin_mcap_b": stablecoins.get("total_mcap_b") if stablecoins else None,
        "tvl_b": tvl.get("total_tvl_b") if tvl else None,
    }
    history.append(snapshot)

    result = {
        "aggregate": {
            "score": agg_score,
            "bias": agg_bias,
            "strength": agg_strength,
            "signals": [{"bias": s["bias"], "score": s["score"], "reason": s["reason"]} for s in signals],
        },
        "stablecoins": stablecoins,
        "tvl": tvl,
        "dex_volume": dex,
        "btc_onchain": btc,
        "eth_onchain": eth,
        "yields": yields,
        "history": history,
    }

    save_data("onchain.json", result)

    log.info(f"AGGREGATE: score={agg_score} bias={agg_bias} strength={agg_strength}")
    log.info("On-chain agent complete.")
    return result


if __name__ == "__main__":
    run()
