"""CryptoAgent v2 — Data Collection Layer.

Sources (all free, all work from US GitHub Actions runners):
1. OKX (funding rates, open interest, taker volume, CVD)
2. TradingView (technical analysis consensus via tradingview_ta)
3. DeFi Llama (stablecoin flows — macro money indicator)
4. CryptoPanic (news sentiment)
5. Alternative.me (Fear & Greed Index)
6. Deribit (options max pain, put/call ratio)

Every function has:
- 90-second in-memory caching
- 3-retry with exponential backoff
- Graceful degradation (returns None on failure, never crashes)
"""
import logging
import time
from functools import wraps

import requests

import config

log = logging.getLogger("data")

# ─── Caching Infrastructure ──────────────────────────────────────────

_cache_store = {}
DEFAULT_CACHE_TTL = 90  # 90 seconds


def cached(ttl=DEFAULT_CACHE_TTL):
    """Decorator for in-memory caching with configurable TTL."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            now = time.time()
            if cache_key in _cache_store:
                entry = _cache_store[cache_key]
                if now - entry["ts"] < ttl:
                    return entry["data"]
            result = func(*args, **kwargs)
            if result is not None:
                _cache_store[cache_key] = {"data": result, "ts": now}
            return result
        return wrapper
    return decorator


def _retry_request(method, url, params=None, timeout=10, retries=3):
    """HTTP request with exponential backoff retry.

    Returns requests.Response or None on total failure.
    """
    for attempt in range(retries):
        try:
            r = requests.request(method, url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            log.debug(f"HTTP {r.status_code} from {url} (attempt {attempt+1})")
        except requests.exceptions.Timeout:
            log.debug(f"Timeout from {url} (attempt {attempt+1})")
        except Exception as e:
            log.debug(f"Request error {url} (attempt {attempt+1}): {e}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


# ─── OKX Instrument Mapping ──────────────────────────────────────────

OKX_MAP = {
    "BTC-USD": {"swap": "BTC-USDT-SWAP", "ccy": "BTC"},
    "ETH-USD": {"swap": "ETH-USDT-SWAP", "ccy": "ETH"},
    "SOL-USD": {"swap": "SOL-USDT-SWAP", "ccy": "SOL"},
}

# TradingView symbol mapping
TV_MAP = {
    "BTC-USD": {"symbol": "BTCUSD", "exchange": "COINBASE"},
    "ETH-USD": {"symbol": "ETHUSD", "exchange": "COINBASE"},
    "SOL-USD": {"symbol": "SOLUSD", "exchange": "COINBASE"},
}


# ═══════════════════════════════════════════════════════════════════════
# 1. OKX — Funding Rates, Open Interest, Taker Volume, CVD
# ═══════════════════════════════════════════════════════════════════════

@cached(ttl=90)
def get_funding_rate(pair):
    """Get funding rates from OKX.

    Funding > 0.01% = longs paying (overleveraged longs, dump risk)
    Funding < -0.01% = shorts paying (overleveraged shorts, squeeze incoming)
    """
    info = OKX_MAP.get(pair)
    if not info:
        return None

    # Get history
    r = _retry_request("GET", "https://www.okx.com/api/v5/public/funding-rate-history",
                       params={"instId": info["swap"], "limit": 10}, timeout=8)
    if not r:
        return None

    data = r.json().get("data", [])
    if not data:
        return None

    rates = [float(d["realizedRate"]) for d in data if d.get("realizedRate")]
    if not rates:
        rates = [float(d["fundingRate"]) for d in data if d.get("fundingRate")]
    if not rates:
        return None

    # Get current predicted rate
    r2 = _retry_request("GET", "https://www.okx.com/api/v5/public/funding-rate",
                        params={"instId": info["swap"]}, timeout=8)
    current_rate = 0
    if r2:
        curr_data = r2.json().get("data", [])
        if curr_data:
            current_rate = float(curr_data[0].get("fundingRate", 0))

    if not current_rate and rates:
        current_rate = rates[0]

    avg_3 = sum(rates[:3]) / min(3, len(rates))

    # Proper streak counting
    pos_streak = 0
    for rate in rates:
        if rate > 0:
            pos_streak += 1
        else:
            break

    neg_streak = 0
    for rate in rates:
        if rate < 0:
            neg_streak += 1
        else:
            break

    signal = _funding_signal(current_rate, avg_3, pos_streak, neg_streak)

    return {
        "current": round(current_rate * 100, 4),
        "avg_3": round(avg_3 * 100, 4),
        "history": [round(rate * 100, 4) for rate in rates[:10]],
        "positive_streak": pos_streak,
        "negative_streak": neg_streak,
        "signal": signal,
    }


def _funding_signal(current, avg_3, pos_streak, neg_streak):
    """Interpret funding rate into a directional signal."""
    if current < -0.0001 and neg_streak >= 2:
        return {"bias": "bullish", "strength": "strong",
                "reason": f"FR {current*100:.4f}% neg x{neg_streak} = short squeeze"}
    if current < -0.00005:
        return {"bias": "bullish", "strength": "moderate",
                "reason": f"FR negative {current*100:.4f}%"}
    if current > 0.0003 and pos_streak >= 2:
        return {"bias": "bearish", "strength": "strong",
                "reason": f"FR {current*100:.4f}% pos x{pos_streak} = overleveraged longs"}
    if current > 0.00015:
        return {"bias": "bearish", "strength": "moderate",
                "reason": f"FR elevated {current*100:.4f}%"}
    return {"bias": "neutral", "strength": "none", "reason": "Funding normal"}


@cached(ttl=90)
def get_open_interest(pair):
    """Get open interest from OKX."""
    info = OKX_MAP.get(pair)
    if not info:
        return None

    r = _retry_request("GET", "https://www.okx.com/api/v5/public/open-interest",
                       params={"instType": "SWAP", "instFamily": f"{info['ccy']}-USDT"},
                       timeout=8)
    if not r:
        return None

    data = r.json().get("data", [])
    if not data:
        return None

    current_oi_usd = float(data[0].get("oiUsd", 0))
    current_oi = float(data[0].get("oi", 0))

    return {
        "current": current_oi,
        "current_usd": current_oi_usd,
        "change_2h_pct": 0,
        "rising": False,
        "falling": False,
    }


@cached(ttl=90)
def get_taker_volume(pair):
    """Get taker buy/sell volume from OKX.

    Buy ratio > 1.15 = aggressive buyers (bullish)
    Buy ratio < 0.85 = aggressive sellers (bearish)
    """
    info = OKX_MAP.get(pair)
    if not info:
        return None

    r = _retry_request("GET", "https://www.okx.com/api/v5/rubik/stat/taker-volume",
                       params={"ccy": info["ccy"], "instType": "CONTRACTS", "period": "5m"},
                       timeout=8)
    if not r:
        return None

    data = r.json().get("data", [])
    if not data or len(data) < 6:
        return None

    ratios = []
    for d in data[:12]:
        buy_vol = float(d[1])
        sell_vol = float(d[2])
        ratio = buy_vol / sell_vol if sell_vol > 0 else 1.0
        ratios.append(ratio)

    current = ratios[0] if ratios else 1.0
    avg = sum(ratios) / len(ratios) if ratios else 1.0

    if len(ratios) >= 3:
        recent = ratios[:3]
        trend = ("buyers_increasing" if all(recent[i] < recent[i + 1] for i in range(2)) else
                 "sellers_increasing" if all(recent[i] > recent[i + 1] for i in range(2)) else "mixed")
    else:
        trend = "mixed"

    return {
        "current": round(current, 3),
        "avg_1h": round(avg, 3),
        "trend": trend,
        "aggressive_buyers": current > 1.15,
        "aggressive_sellers": current < 0.85,
    }


@cached(ttl=90)
def get_cvd_divergence(pair):
    """CVD (Cumulative Volume Delta) divergence from OKX taker data.

    Price down + aggressive buying = ACCUMULATION (bullish)
    Price up + aggressive selling = DISTRIBUTION (bearish)
    """
    info = OKX_MAP.get(pair)
    if not info:
        return None

    # Get taker volumes (5min intervals)
    r = _retry_request("GET", "https://www.okx.com/api/v5/rubik/stat/taker-volume",
                       params={"ccy": info["ccy"], "instType": "CONTRACTS", "period": "5m"},
                       timeout=8)
    if not r:
        return None

    taker_data = r.json().get("data", [])
    if len(taker_data) < 12:
        return None

    # Build CVD
    cvd = []
    cumulative = 0
    for d in reversed(taker_data[:24]):  # Oldest first
        buy_vol = float(d[1])
        sell_vol = float(d[2])
        delta = buy_vol - sell_vol
        cumulative += delta
        cvd.append(cumulative)

    # Get price from OKX candles
    r2 = _retry_request("GET", "https://www.okx.com/api/v5/market/candles",
                        params={"instId": info["swap"], "bar": "5m", "limit": "24"},
                        timeout=8)
    if not r2:
        return None

    candle_data = r2.json().get("data", [])
    prices = [float(c[4]) for c in reversed(candle_data)]  # Close prices, oldest first

    if len(cvd) < 12 or len(prices) < 12:
        return None

    # Compare slopes
    mid = len(cvd) // 2
    cvd_slope = cvd[-1] - cvd[mid]
    price_mid = min(mid, len(prices) - 1)
    price_slope = prices[-1] - prices[price_mid]

    bearish_div = price_slope > 0 and cvd_slope < 0
    bullish_div = price_slope < 0 and cvd_slope > 0

    return {
        "cvd_current": round(cumulative, 2),
        "cvd_slope": round(cvd_slope, 2),
        "price_slope": round(price_slope, 2),
        "bearish_divergence": bearish_div,
        "bullish_divergence": bullish_div,
        "signal": "bullish" if bullish_div else "bearish" if bearish_div else "neutral",
    }


def collect_okx(pair):
    """Collect all OKX data for a pair."""
    result = {}

    funding = get_funding_rate(pair)
    if funding:
        result["funding"] = funding

    oi = get_open_interest(pair)
    if oi:
        result["open_interest"] = oi

    taker = get_taker_volume(pair)
    if taker:
        result["taker_ratio"] = taker

    cvd = get_cvd_divergence(pair)
    if cvd:
        result["cvd"] = cvd

    result["composite"] = _composite_signal(result)
    return result


def _composite_signal(data):
    """Combine all OKX signals into composite score (-100 to +100)."""
    score = 0
    factors = []

    # Funding rate (strongest single signal)
    funding = data.get("funding", {})
    fsig = funding.get("signal", {})
    if fsig.get("bias") == "bullish":
        s = 30 if fsig.get("strength") == "strong" else 15
        score += s
        factors.append(f"FR={funding.get('current', 0):.3f}%")
    elif fsig.get("bias") == "bearish":
        s = 30 if fsig.get("strength") == "strong" else 15
        score -= s
        factors.append(f"FR={funding.get('current', 0):.3f}%")

    # Taker volume
    taker = data.get("taker_ratio", {})
    if taker.get("aggressive_buyers"):
        score += 15
        factors.append(f"taker={taker.get('current', 0):.2f}")
    elif taker.get("aggressive_sellers"):
        score -= 15
        factors.append(f"taker={taker.get('current', 0):.2f}")

    # CVD divergence
    cvd = data.get("cvd", {})
    if cvd.get("bullish_divergence"):
        score += 25
        factors.append("CVD_bull")
    elif cvd.get("bearish_divergence"):
        score -= 25
        factors.append("CVD_bear")

    score = max(-100, min(100, score))
    bias = "bullish" if score > 20 else "bearish" if score < -20 else "neutral"
    return {"score": score, "bias": bias, "factors": factors}


# ═══════════════════════════════════════════════════════════════════════
# 2. TradingView — Technical Analysis Consensus
# ═══════════════════════════════════════════════════════════════════════

@cached(ttl=90)
def get_tradingview_analysis(pair, interval="1h"):
    """Get TradingView technical analysis summary.

    Returns oscillators + moving averages consensus (BUY/SELL/NEUTRAL).
    Uses the tradingview_ta library.

    interval: "1m", "5m", "15m", "1h", "4h", "1d", "1W", "1M"
    """
    if not config.TV_ENABLED:
        return None

    tv_info = TV_MAP.get(pair)
    if not tv_info:
        return None

    interval_map = {
        "1m": "INTERVAL_1_MINUTE",
        "5m": "INTERVAL_5_MINUTES",
        "15m": "INTERVAL_15_MINUTES",
        "1h": "INTERVAL_1_HOUR",
        "4h": "INTERVAL_4_HOURS",
        "1d": "INTERVAL_1_DAY",
        "1W": "INTERVAL_1_WEEK",
        "1M": "INTERVAL_1_MONTH",
    }

    try:
        from tradingview_ta import TA_Handler, Interval

        tv_interval = getattr(Interval, interval_map.get(interval, "INTERVAL_1_HOUR"), Interval.INTERVAL_1_HOUR)

        handler = TA_Handler(
            symbol=tv_info["symbol"],
            exchange=tv_info["exchange"],
            screener="crypto",
            interval=tv_interval,
        )
        analysis = handler.get_analysis()

        summary = analysis.summary
        oscillators = analysis.oscillators
        moving_averages = analysis.moving_averages

        # Extract recommendation
        rec = summary.get("RECOMMENDATION", "NEUTRAL")

        # Count signals
        buy_count = summary.get("BUY", 0)
        sell_count = summary.get("SELL", 0)
        neutral_count = summary.get("NEUTRAL", 0)
        total = buy_count + sell_count + neutral_count

        # Oscillator details
        osc_rec = oscillators.get("RECOMMENDATION", "NEUTRAL")
        ma_rec = moving_averages.get("RECOMMENDATION", "NEUTRAL")

        # Convert to our bias format
        bias_map = {
            "STRONG_BUY": "bullish",
            "BUY": "bullish",
            "NEUTRAL": "neutral",
            "SELL": "bearish",
            "STRONG_SELL": "bearish",
        }
        strength_map = {
            "STRONG_BUY": "strong",
            "BUY": "moderate",
            "NEUTRAL": "none",
            "SELL": "moderate",
            "STRONG_SELL": "strong",
        }

        # Extract key indicator values
        indicators = analysis.indicators
        rsi = indicators.get("RSI", None)
        stoch_k = indicators.get("Stoch.K", None)
        cci = indicators.get("CCI20", None)
        macd_val = indicators.get("MACD.macd", None)
        macd_sig = indicators.get("MACD.signal", None)
        adx = indicators.get("ADX", None)
        ao = indicators.get("AO", None)

        return {
            "recommendation": rec,
            "bias": bias_map.get(rec, "neutral"),
            "strength": strength_map.get(rec, "none"),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "neutral_signals": neutral_count,
            "total_signals": total,
            "oscillators_rec": osc_rec,
            "moving_averages_rec": ma_rec,
            "interval": interval,
            "indicators": {
                "rsi": round(rsi, 2) if rsi is not None else None,
                "stoch_k": round(stoch_k, 2) if stoch_k is not None else None,
                "cci": round(cci, 2) if cci is not None else None,
                "macd": round(macd_val, 4) if macd_val is not None else None,
                "macd_signal": round(macd_sig, 4) if macd_sig is not None else None,
                "adx": round(adx, 2) if adx is not None else None,
                "ao": round(ao, 4) if ao is not None else None,
            },
        }

    except ImportError:
        log.warning("tradingview_ta not installed — pip install tradingview_ta")
        return None
    except Exception as e:
        log.debug(f"TradingView analysis error for {pair} ({interval}): {e}")
        return None


@cached(ttl=90)
def get_tradingview_multi_timeframe(pair):
    """Get TradingView analysis across multiple timeframes.

    Multi-timeframe alignment is a strong signal:
    - All timeframes BUY = high confidence long
    - All timeframes SELL = high confidence short
    - Mixed = choppy/ranging
    """
    timeframes = ["15m", "1h", "4h", "1d"]
    results = {}
    alignment_score = 0

    for tf in timeframes:
        analysis = get_tradingview_analysis(pair, interval=tf)
        if analysis:
            results[tf] = analysis
            # Score: +1 for bullish, -1 for bearish, 0 for neutral
            if analysis["bias"] == "bullish":
                weight = 2 if analysis["strength"] == "strong" else 1
                alignment_score += weight
            elif analysis["bias"] == "bearish":
                weight = 2 if analysis["strength"] == "strong" else 1
                alignment_score -= weight

    total_tf = len(results)
    if total_tf == 0:
        return None

    # Normalize score to -100 to +100
    max_possible = total_tf * 2  # All strong in one direction
    normalized = int(alignment_score / max_possible * 100) if max_possible > 0 else 0

    if normalized > 40:
        mtf_bias = "bullish"
    elif normalized < -40:
        mtf_bias = "bearish"
    else:
        mtf_bias = "neutral"

    return {
        "timeframes": results,
        "alignment_score": normalized,
        "bias": mtf_bias,
        "aligned": abs(normalized) > 60,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. DeFi Llama — Stablecoin Flows
# ═══════════════════════════════════════════════════════════════════════

@cached(ttl=1800)  # 30 minute cache — this data changes slowly
def get_stablecoin_flows():
    """Stablecoin market cap changes from DeFi Llama.

    Stablecoin mcap rising = money flowing INTO crypto (bullish)
    Stablecoin mcap falling = money flowing OUT (bearish)

    This is the macro indicator that catches big moves days in advance.
    """
    r = _retry_request("GET", "https://stablecoins.llama.fi/stablecoins?includePrices=false",
                       timeout=10)
    if not r:
        return None

    data = r.json()
    stables = data.get("peggedAssets", [])

    target_names = {"Tether", "USD Coin", "Dai", "BUSD", "First Digital USD", "USDS"}
    total_mcap = 0
    breakdown = {}

    for s in stables:
        name = s.get("name", "")
        if name in target_names:
            asset_mcap = 0
            chains = s.get("chainCirculating", {})
            for chain, data_chain in chains.items():
                current = data_chain.get("current", {})
                asset_mcap += current.get("peggedUSD", 0)
            total_mcap += asset_mcap
            if asset_mcap > 0:
                breakdown[name] = round(asset_mcap / 1e9, 2)

    return {
        "total_mcap_b": round(total_mcap / 1e9, 2),
        "breakdown": breakdown,
    }


@cached(ttl=1800)
def get_stablecoin_history():
    """Get stablecoin total market cap history for trend analysis."""
    r = _retry_request("GET", "https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1",
                       timeout=10)
    if not r:
        return None

    data = r.json()
    if not data or len(data) < 7:
        return None

    # Get last 30 days of data points
    recent = data[-30:]
    mcaps = []
    for d in recent:
        total = sum(v.get("peggedUSD", 0) for v in d.get("totalCirculating", {}).values())
        mcaps.append(total)

    if len(mcaps) < 7:
        return None

    current = mcaps[-1]
    week_ago = mcaps[-7]
    month_ago = mcaps[0]

    week_change = ((current - week_ago) / week_ago * 100) if week_ago else 0
    month_change = ((current - month_ago) / month_ago * 100) if month_ago else 0

    return {
        "current_b": round(current / 1e9, 2),
        "week_change_pct": round(week_change, 2),
        "month_change_pct": round(month_change, 2),
        "trend": "inflow" if week_change > 0.5 else "outflow" if week_change < -0.5 else "flat",
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. CryptoPanic — News Sentiment
# ═══════════════════════════════════════════════════════════════════════

@cached(ttl=300)  # 5 minute cache
def get_news_headlines(currencies=None, limit=10):
    """Fetch recent crypto news headlines from CryptoPanic.

    Returns list of dicts with title, source, url, sentiment.
    """
    params = {"kind": "news", "public": "true"}
    if config.CRYPTOPANIC_API_KEY:
        params["auth_token"] = config.CRYPTOPANIC_API_KEY
    if currencies:
        params["currencies"] = ",".join(c.lower() for c in currencies)

    r = _retry_request("GET", "https://cryptopanic.com/api/free/v1/posts/",
                       params=params, timeout=10)
    if not r:
        return _fallback_news()

    data = r.json()
    results = []
    for post in data.get("results", [])[:limit]:
        votes = post.get("votes", {})
        pos = votes.get("positive", 0) + votes.get("liked", 0)
        neg = votes.get("negative", 0) + votes.get("disliked", 0)
        total = pos + neg
        sentiment = "neutral"
        if total > 0:
            ratio = pos / total
            sentiment = "bullish" if ratio > 0.6 else "bearish" if ratio < 0.4 else "neutral"

        results.append({
            "title": post.get("title", ""),
            "source": post.get("source", {}).get("title", ""),
            "url": post.get("url", ""),
            "published": post.get("published_at", ""),
            "sentiment": sentiment,
            "votes": {"positive": pos, "negative": neg},
        })

    return results[:limit]


def _fallback_news():
    """Fallback news via CoinGecko trending when CryptoPanic fails."""
    r = _retry_request("GET", "https://api.coingecko.com/api/v3/search/trending", timeout=10)
    if not r:
        return []

    data = r.json()
    results = []
    for coin in data.get("coins", [])[:5]:
        item = coin.get("item", {})
        results.append({
            "title": f"{item.get('name', '')} ({item.get('symbol', '')}) trending - "
                     f"market cap rank #{item.get('market_cap_rank', '?')}",
            "source": "CoinGecko",
            "url": "",
            "published": "",
            "sentiment": "bullish",
            "votes": {"positive": 1, "negative": 0},
        })
    return results


def summarize_sentiment(headlines):
    """Aggregate news sentiment into a summary."""
    if not headlines:
        return {"overall": "neutral", "bullish": 0, "bearish": 0, "neutral": 0, "headlines": []}

    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for h in headlines:
        counts[h.get("sentiment", "neutral")] += 1

    total = len(headlines)
    if counts["bullish"] > counts["bearish"] and counts["bullish"] / total > 0.4:
        overall = "bullish"
    elif counts["bearish"] > counts["bullish"] and counts["bearish"] / total > 0.4:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "bullish": counts["bullish"],
        "bearish": counts["bearish"],
        "neutral": counts["neutral"],
        "headlines": [h["title"] for h in headlines[:5]],
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. Alternative.me — Fear & Greed Index
# ═══════════════════════════════════════════════════════════════════════

@cached(ttl=3600)  # 1 hour cache — updates once daily
def get_fear_greed_index(days=7):
    """Fetch Fear & Greed Index from alternative.me.

    FGI <= 25 (Extreme Fear) = contrarian BUY signal
    FGI >= 75 (Extreme Greed) = contrarian SELL signal
    """
    r = _retry_request("GET", f"https://api.alternative.me/fng/?limit={days}&format=json",
                       timeout=10)
    if not r:
        return _default_fgi()

    data = r.json().get("data", [])
    if not data:
        return _default_fgi()

    result = {
        "current": int(data[0]["value"]),
        "classification": data[0].get("value_classification", "Neutral"),
        "history": [int(d["value"]) for d in data],
        "avg_7d": round(sum(int(d["value"]) for d in data) / len(data), 1),
        "consecutive_fear_days": 0,
        "consecutive_greed_days": 0,
    }

    for d in data:
        if int(d["value"]) < 30:
            result["consecutive_fear_days"] += 1
        else:
            break

    for d in data:
        if int(d["value"]) > 70:
            result["consecutive_greed_days"] += 1
        else:
            break

    log.info(f"FGI: {result['current']} ({result['classification']}) | "
             f"7d avg: {result['avg_7d']} | Fear streak: {result['consecutive_fear_days']}d")
    return result


def _default_fgi():
    """Default FGI when API is unavailable."""
    return {
        "current": 50, "classification": "Neutral", "history": [],
        "avg_7d": 50, "consecutive_fear_days": 0, "consecutive_greed_days": 0,
    }


def fgi_signal(fgi_data, price, rsi=None, bb_pct=None):
    """Generate contrarian signal from Fear & Greed + technicals.

    Extreme fear + oversold technicals = strongest buy signal.
    Extreme greed + overbought technicals = strongest sell signal.
    """
    fgi = fgi_data.get("current", 50)
    consec_fear = fgi_data.get("consecutive_fear_days", 0)
    consec_greed = fgi_data.get("consecutive_greed_days", 0)

    if not price:
        return None

    # ACCUMULATION BUY: extreme fear
    if fgi <= 25 and consec_fear >= 2:
        confidence = 0.60
        if fgi <= 10:
            confidence = 0.80
        elif fgi <= 15:
            confidence = 0.75
        elif fgi <= 20:
            confidence = 0.70

        if rsi is not None and rsi < 30:
            confidence += 0.05
        if bb_pct is not None and bb_pct < 0.1:
            confidence += 0.05

        return {
            "action": "buy",
            "confidence": min(confidence, 0.90),
            "strategy": "fgi_contrarian",
            "reasoning": f"FGI contrarian BUY: FGI={fgi} ({consec_fear}d fear)"
                         + (f", RSI={rsi:.0f}" if rsi else ""),
        }

    # DISTRIBUTION SELL: extreme greed
    if fgi >= 80 and consec_greed >= 4:
        confidence = 0.70
        if fgi >= 90:
            confidence = 0.85

        if bb_pct is not None and bb_pct > 0.85:
            confidence += 0.05

        return {
            "action": "sell",
            "confidence": min(confidence, 0.90),
            "strategy": "fgi_contrarian",
            "reasoning": f"FGI contrarian SELL: FGI={fgi} ({consec_greed}d greed)"
                         + (f", BB%={bb_pct:.2f}" if bb_pct is not None else ""),
        }

    return None


# ═══════════════════════════════════════════════════════════════════════
# 6. Deribit — Options Max Pain & Put/Call Ratio
# ═══════════════════════════════════════════════════════════════════════

@cached(ttl=600)  # 10 minute cache — options data doesn't change fast
def get_options_data(currency="BTC"):
    """Get options max pain and put/call ratio from Deribit.

    Max pain = price where option sellers pay least — price gravitates here before expiry.
    PC ratio > 1.2 = fearful (contrarian bullish)
    PC ratio < 0.6 = greedy (contrarian bearish)
    """
    r = _retry_request("GET",
                       "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                       params={"currency": currency, "kind": "option"},
                       timeout=10)
    if not r:
        return None

    data = r.json().get("result", [])
    if not data:
        return None

    strikes = {}
    total_call_oi = 0
    total_put_oi = 0

    for opt in data:
        name = opt.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) < 4:
            continue
        try:
            strike = int(parts[2])
            opt_type = parts[3]
        except (ValueError, IndexError):
            continue

        oi = opt.get("open_interest", 0)
        if strike not in strikes:
            strikes[strike] = {"call_oi": 0, "put_oi": 0}
        if opt_type == "C":
            strikes[strike]["call_oi"] += oi
            total_call_oi += oi
        else:
            strikes[strike]["put_oi"] += oi
            total_put_oi += oi

    if not strikes:
        return None

    # Calculate max pain
    min_pain = float("inf")
    max_pain_strike = 0
    for test_price in strikes:
        total_pain = 0
        for strike, oi_data in strikes.items():
            if test_price > strike:
                total_pain += (test_price - strike) * oi_data["call_oi"]
            if test_price < strike:
                total_pain += (strike - test_price) * oi_data["put_oi"]
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_price

    pc_ratio = total_put_oi / max(total_call_oi, 1)

    # Find highest OI strikes (support/resistance levels)
    sorted_strikes = sorted(strikes.items(), key=lambda x: x[1]["call_oi"] + x[1]["put_oi"], reverse=True)
    top_strikes = [{"strike": s[0], "call_oi": s[1]["call_oi"], "put_oi": s[1]["put_oi"]}
                   for s in sorted_strikes[:5]]

    return {
        "max_pain": max_pain_strike,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "pc_ratio": round(pc_ratio, 3),
        "sentiment": "fearful" if pc_ratio > 1.2 else "greedy" if pc_ratio < 0.6 else "neutral",
        "top_strikes": top_strikes,
    }


@cached(ttl=600)
def get_deribit_volatility_index(currency="BTC"):
    """Get Deribit Volatility Index (DVOL) — implied volatility of the market.

    High DVOL = market expects large moves (opportunity for entries after spike)
    Low DVOL = market is calm (range-bound, tighter stops)
    """
    r = _retry_request("GET",
                       "https://www.deribit.com/api/v2/public/get_volatility_index_data",
                       params={"currency": currency, "resolution": "3600", "start_timestamp": int((time.time() - 86400) * 1000),
                               "end_timestamp": int(time.time() * 1000)},
                       timeout=10)
    if not r:
        return None

    data = r.json().get("result", {}).get("data", [])
    if not data:
        return None

    # Data format: [[timestamp, open, high, low, close], ...]
    current_vol = data[-1][4] if data else 0
    vols = [d[4] for d in data]
    avg_24h = sum(vols) / len(vols) if vols else 0

    high_24h = max(vols) if vols else 0
    low_24h = min(vols) if vols else 0

    return {
        "current": round(current_vol, 2),
        "avg_24h": round(avg_24h, 2),
        "high_24h": round(high_24h, 2),
        "low_24h": round(low_24h, 2),
        "elevated": current_vol > avg_24h * 1.2,
        "suppressed": current_vol < avg_24h * 0.8,
    }


# ═══════════════════════════════════════════════════════════════════════
# Liquidation Magnet Estimation (derived from candle data)
# ═══════════════════════════════════════════════════════════════════════

def estimate_liquidation_magnets(candles_1h, current_price):
    """Estimate liquidation cluster locations — price hunts these levels.

    Derived from swing highs/lows and common leverage levels.
    """
    if not candles_1h or len(candles_1h) < 24 or not current_price:
        return None

    lows = [c["low"] for c in candles_1h[-24:]]
    highs = [c["high"] for c in candles_1h[-24:]]
    swing_low = min(lows)
    swing_high = max(highs)

    long_liqs = {}
    for lev in [5, 10, 25, 50]:
        liq_price = swing_high * (1 - 0.9 / lev)
        if liq_price < current_price:
            long_liqs[lev] = round(liq_price, 2)

    short_liqs = {}
    for lev in [5, 10, 25, 50]:
        liq_price = swing_low * (1 + 0.9 / lev)
        if liq_price > current_price:
            short_liqs[lev] = round(liq_price, 2)

    nearest_long = max(long_liqs.values()) if long_liqs else 0
    nearest_short = min(short_liqs.values()) if short_liqs else float("inf")

    dist_long = abs(current_price - nearest_long) / current_price if nearest_long else 1
    dist_short = abs(nearest_short - current_price) / current_price if nearest_short < float("inf") else 1

    if dist_long < dist_short and dist_long < 0.03:
        return {
            "bias": "bearish",
            "magnet": nearest_long,
            "distance_pct": round(dist_long * 100, 2),
            "reason": f"Long liqs at ${nearest_long:,.0f} ({dist_long * 100:.1f}% away)",
        }
    elif dist_short < dist_long and dist_short < 0.03:
        return {
            "bias": "bullish",
            "magnet": nearest_short,
            "distance_pct": round(dist_short * 100, 2),
            "reason": f"Short liqs at ${nearest_short:,.0f} ({dist_short * 100:.1f}% away)",
        }

    return {"bias": "neutral", "reason": "No nearby liquidation magnets"}


# ═══════════════════════════════════════════════════════════════════════
# Master Collection — All Sources for a Trading Pair
# ═══════════════════════════════════════════════════════════════════════

def collect_all(pair):
    """Collect ALL data sources for a trading pair.

    Returns a comprehensive dict with data from every source.
    Individual source failures don't prevent other sources from loading.
    """
    result = {}
    currency = pair.split("-")[0]

    # 1. OKX derivatives data
    okx = collect_okx(pair)
    if okx:
        result["okx"] = okx

    # 2. TradingView technical analysis
    tv = get_tradingview_multi_timeframe(pair)
    if tv:
        result["tradingview"] = tv

    # 3. DeFi Llama stablecoin flows
    stables = get_stablecoin_flows()
    if stables:
        result["stablecoins"] = stables
    stable_history = get_stablecoin_history()
    if stable_history:
        result["stablecoin_trend"] = stable_history

    # 4. News sentiment
    headlines = get_news_headlines(currencies=[currency])
    if headlines:
        result["news"] = summarize_sentiment(headlines)

    # 5. Fear & Greed Index
    fgi = get_fear_greed_index()
    if fgi:
        result["fear_greed"] = fgi

    # 6. Deribit options (BTC and ETH only — Deribit doesn't list SOL options)
    if currency in ("BTC", "ETH"):
        options = get_options_data(currency)
        if options:
            result["options"] = options
        dvol = get_deribit_volatility_index(currency)
        if dvol:
            result["volatility_index"] = dvol

    # Master signal aggregation
    result["master_signal"] = _master_signal(result)

    return result


def _master_signal(data):
    """Aggregate all data sources into a single directional signal.

    Score: -100 (max bearish) to +100 (max bullish)
    Each source contributes weighted points based on reliability.
    """
    score = 0
    factors = []

    # OKX composite (weight: 30 — derivatives data is the most actionable)
    okx = data.get("okx", {})
    okx_comp = okx.get("composite", {})
    okx_score = okx_comp.get("score", 0)
    score += int(okx_score * 0.30)
    if okx_comp.get("factors"):
        factors.extend(okx_comp["factors"])

    # TradingView multi-timeframe (weight: 25 — technical consensus)
    tv = data.get("tradingview", {})
    tv_score = tv.get("alignment_score", 0)
    score += int(tv_score * 0.25)
    if tv.get("bias") != "neutral":
        factors.append(f"TV={tv.get('bias', 'n/a')}({tv_score})")

    # Fear & Greed (weight: 15 — contrarian macro)
    fgi = data.get("fear_greed", {})
    fgi_val = fgi.get("current", 50)
    if fgi_val <= 20:
        score += 15
        factors.append(f"FGI={fgi_val}(extreme_fear)")
    elif fgi_val <= 35:
        score += 8
        factors.append(f"FGI={fgi_val}(fear)")
    elif fgi_val >= 80:
        score -= 15
        factors.append(f"FGI={fgi_val}(extreme_greed)")
    elif fgi_val >= 65:
        score -= 8
        factors.append(f"FGI={fgi_val}(greed)")

    # News sentiment (weight: 10 — crowd noise, contrarian)
    news = data.get("news", {})
    if news.get("overall") == "bullish":
        score += 5
        factors.append("news=bullish")
    elif news.get("overall") == "bearish":
        score -= 5
        factors.append("news=bearish")

    # Options data (weight: 10 — institutional positioning)
    options = data.get("options", {})
    if options.get("sentiment") == "fearful":
        score += 10
        factors.append(f"PC={options.get('pc_ratio', 0):.2f}")
    elif options.get("sentiment") == "greedy":
        score -= 10
        factors.append(f"PC={options.get('pc_ratio', 0):.2f}")

    # Stablecoin flows (weight: 10 — macro money flow)
    stable_trend = data.get("stablecoin_trend", {})
    if stable_trend.get("trend") == "inflow":
        score += 10
        factors.append(f"stables_inflow({stable_trend.get('week_change_pct', 0):.1f}%)")
    elif stable_trend.get("trend") == "outflow":
        score -= 10
        factors.append(f"stables_outflow({stable_trend.get('week_change_pct', 0):.1f}%)")

    score = max(-100, min(100, score))
    if score > 25:
        bias = "bullish"
    elif score < -25:
        bias = "bearish"
    else:
        bias = "neutral"

    strength = "strong" if abs(score) > 60 else "moderate" if abs(score) > 35 else "weak"

    return {
        "score": score,
        "bias": bias,
        "strength": strength,
        "factors": factors,
        "source_count": len([k for k in data.keys() if k != "master_signal"]),
    }


def clear_cache():
    """Clear all cached data. Useful for forcing fresh data collection."""
    global _cache_store
    _cache_store.clear()
    log.info("Data cache cleared")
