#!/usr/bin/env python3
"""Intel Sub-Agent: Sentiment Scraping.

Sources (all free, no API keys required for basic access):
1. Alternative.me Fear & Greed Index — free, 60 req/min
2. CryptoPanic public API — free, no auth for public posts
3. Reddit (old.reddit.com JSON) — free, no auth needed
4. CoinGecko trending — free, 10-30 req/min
5. LunarCrush open API — free tier

Runs: Every hour via GitHub Actions
Output: intel/data/sentiment.json
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import retry_get, load_data, save_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("sentiment")

COINS = ["BTC", "ETH", "SOL"]
SUBREDDITS = ["cryptocurrency", "bitcoin", "ethtrader", "solana"]

# Keyword sentiment scoring
BULLISH_WORDS = {
    "moon", "pump", "bullish", "buy", "long", "breakout", "ath",
    "accumulate", "rally", "surge", "soar", "rocket", "green",
    "dip buying", "btfd", "undervalued", "fomo", "send it",
}
BEARISH_WORDS = {
    "crash", "dump", "bearish", "sell", "short", "collapse", "liquidat",
    "scam", "rug", "bubble", "overvalued", "dead", "rekt", "capitulat",
    "blood", "tank", "plunge", "fear", "panic",
}


def fetch_fear_greed():
    """Fetch Fear & Greed Index from alternative.me."""
    log.info("Fetching Fear & Greed Index...")
    r = retry_get("https://api.alternative.me/fng/?limit=30&format=json")
    if not r:
        return None

    data = r.json().get("data", [])
    if not data:
        return None

    history = []
    for d in data:
        history.append({
            "value": int(d["value"]),
            "classification": d.get("value_classification", ""),
            "timestamp": int(d.get("timestamp", 0)),
        })

    current = history[0] if history else {"value": 50, "classification": "Neutral"}

    # Calculate trends
    vals = [h["value"] for h in history]
    avg_7d = sum(vals[:7]) / min(7, len(vals)) if vals else 50
    avg_30d = sum(vals[:30]) / min(30, len(vals)) if vals else 50

    # Consecutive fear/greed days
    fear_streak = 0
    for v in vals:
        if v < 30:
            fear_streak += 1
        else:
            break

    greed_streak = 0
    for v in vals:
        if v > 70:
            greed_streak += 1
        else:
            break

    result = {
        "current": current["value"],
        "classification": current["classification"],
        "avg_7d": round(avg_7d, 1),
        "avg_30d": round(avg_30d, 1),
        "fear_streak": fear_streak,
        "greed_streak": greed_streak,
        "history_30d": vals[:30],
        "trend": "improving" if avg_7d > avg_30d else "declining" if avg_7d < avg_30d else "flat",
    }

    # Generate signal
    if current["value"] <= 20:
        result["signal"] = {"bias": "bullish", "score": 80, "weight": 2.0,
                            "reason": f"Extreme Fear ({current['value']}) — contrarian BUY"}
    elif current["value"] <= 35:
        result["signal"] = {"bias": "bullish", "score": 40, "weight": 1.5,
                            "reason": f"Fear ({current['value']}) — lean bullish"}
    elif current["value"] >= 85:
        result["signal"] = {"bias": "bearish", "score": 80, "weight": 2.0,
                            "reason": f"Extreme Greed ({current['value']}) — contrarian SELL"}
    elif current["value"] >= 70:
        result["signal"] = {"bias": "bearish", "score": 40, "weight": 1.5,
                            "reason": f"Greed ({current['value']}) — lean bearish"}
    else:
        result["signal"] = {"bias": "neutral", "score": 0, "weight": 0.5,
                            "reason": f"Neutral ({current['value']})"}

    log.info(f"FGI: {current['value']} ({current['classification']}) | "
             f"7d avg: {avg_7d:.0f} | streak: fear={fear_streak} greed={greed_streak}")
    return result


def fetch_cryptopanic():
    """Fetch news sentiment from CryptoPanic public API."""
    log.info("Fetching CryptoPanic news...")
    auth_token = os.environ.get("CRYPTOPANIC_API_KEY", "")

    results = []
    for coin in COINS:
        params = {"kind": "news", "public": "true", "currencies": coin.lower()}
        if auth_token:
            params["auth_token"] = auth_token

        r = retry_get("https://cryptopanic.com/api/free/v1/posts/", params=params)
        if not r:
            continue

        data = r.json()
        for post in data.get("results", [])[:15]:
            votes = post.get("votes", {})
            pos = votes.get("positive", 0) + votes.get("liked", 0)
            neg = votes.get("negative", 0) + votes.get("disliked", 0)
            total_votes = pos + neg

            if total_votes > 0:
                ratio = pos / total_votes
                sentiment = "bullish" if ratio > 0.6 else "bearish" if ratio < 0.4 else "neutral"
            else:
                # Keyword analysis fallback
                title = post.get("title", "").lower()
                bull_hits = sum(1 for w in BULLISH_WORDS if w in title)
                bear_hits = sum(1 for w in BEARISH_WORDS if w in title)
                if bull_hits > bear_hits:
                    sentiment = "bullish"
                elif bear_hits > bull_hits:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"

            results.append({
                "title": post.get("title", ""),
                "coin": coin,
                "source": post.get("source", {}).get("title", ""),
                "sentiment": sentiment,
                "votes_pos": pos,
                "votes_neg": neg,
                "published": post.get("published_at", ""),
            })

        # Rate limit respect — 1 sec between requests
        time.sleep(1)

    if not results:
        return None

    # Aggregate per coin
    coin_sentiment = {}
    for coin in COINS:
        coin_posts = [r for r in results if r["coin"] == coin]
        if not coin_posts:
            continue
        bull = sum(1 for p in coin_posts if p["sentiment"] == "bullish")
        bear = sum(1 for p in coin_posts if p["sentiment"] == "bearish")
        total = len(coin_posts)
        ratio = bull / total if total > 0 else 0.5

        coin_sentiment[coin] = {
            "bullish": bull,
            "bearish": bear,
            "neutral": total - bull - bear,
            "total": total,
            "ratio": round(ratio, 3),
            "bias": "bullish" if ratio > 0.55 else "bearish" if ratio < 0.45 else "neutral",
        }

    overall_bull = sum(v["bullish"] for v in coin_sentiment.values())
    overall_bear = sum(v["bearish"] for v in coin_sentiment.values())
    overall_total = sum(v["total"] for v in coin_sentiment.values())
    overall_ratio = overall_bull / overall_total if overall_total > 0 else 0.5

    result = {
        "coins": coin_sentiment,
        "headlines": [{"title": r["title"], "coin": r["coin"], "sentiment": r["sentiment"]}
                      for r in results[:20]],
        "overall": {
            "bullish": overall_bull,
            "bearish": overall_bear,
            "total": overall_total,
            "ratio": round(overall_ratio, 3),
        },
    }

    log.info(f"CryptoPanic: {overall_bull}B/{overall_bear}S/{overall_total}T "
             f"ratio={overall_ratio:.2f}")
    return result


def fetch_reddit_sentiment():
    """Fetch Reddit crypto sentiment from public JSON endpoints.

    Reddit's old.reddit.com/r/<sub>/hot.json works without auth.
    Rate limit: ~60 req/min for unauthenticated. We only make 4 requests.
    """
    log.info("Fetching Reddit sentiment...")
    headers = {"User-Agent": "CryptoAgent/2.0 (research bot)"}
    all_posts = []

    for sub in SUBREDDITS:
        url = f"https://old.reddit.com/r/{sub}/hot.json?limit=25"
        r = retry_get(url, headers=headers, timeout=15)
        if not r:
            log.warning(f"Reddit fetch failed for r/{sub}")
            time.sleep(2)
            continue

        data = r.json().get("data", {}).get("children", [])
        for post_wrap in data:
            post = post_wrap.get("data", {})
            title = post.get("title", "").lower()
            score = post.get("score", 0)
            num_comments = post.get("num_comments", 0)

            # Keyword sentiment analysis
            bull_hits = sum(1 for w in BULLISH_WORDS if w in title)
            bear_hits = sum(1 for w in BEARISH_WORDS if w in title)

            if bull_hits > bear_hits:
                sentiment = "bullish"
            elif bear_hits > bull_hits:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            # Weight by engagement (upvotes + comments)
            engagement = score + num_comments * 2

            all_posts.append({
                "subreddit": sub,
                "title": post.get("title", ""),
                "score": score,
                "comments": num_comments,
                "sentiment": sentiment,
                "engagement": engagement,
            })

        # Rate limit: 2 seconds between subreddits
        time.sleep(2)

    if not all_posts:
        return None

    # Weight sentiment by engagement
    total_engagement = sum(p["engagement"] for p in all_posts) or 1
    weighted_score = 0
    for p in all_posts:
        w = p["engagement"] / total_engagement
        if p["sentiment"] == "bullish":
            weighted_score += w
        elif p["sentiment"] == "bearish":
            weighted_score -= w

    bull = sum(1 for p in all_posts if p["sentiment"] == "bullish")
    bear = sum(1 for p in all_posts if p["sentiment"] == "bearish")

    result = {
        "posts_analyzed": len(all_posts),
        "bullish": bull,
        "bearish": bear,
        "neutral": len(all_posts) - bull - bear,
        "weighted_score": round(weighted_score, 4),
        "top_posts": sorted(all_posts, key=lambda x: x["engagement"], reverse=True)[:10],
    }

    bias = "bullish" if weighted_score > 0.1 else "bearish" if weighted_score < -0.1 else "neutral"
    result["bias"] = bias

    log.info(f"Reddit: {bull}B/{bear}S/{len(all_posts)}T weighted={weighted_score:.3f} => {bias}")
    return result


def fetch_coingecko_trending():
    """Fetch CoinGecko trending coins — market attention indicator."""
    log.info("Fetching CoinGecko trending...")
    r = retry_get("https://api.coingecko.com/api/v3/search/trending")
    if not r:
        return None

    data = r.json()
    coins = []
    for coin_wrap in data.get("coins", [])[:10]:
        item = coin_wrap.get("item", {})
        coins.append({
            "name": item.get("name", ""),
            "symbol": item.get("symbol", ""),
            "market_cap_rank": item.get("market_cap_rank"),
            "score": item.get("score", 0),
        })

    # Check if any of our target coins are trending (strong signal)
    our_coins_trending = [c for c in coins if c["symbol"].upper() in COINS]

    result = {
        "trending": coins,
        "our_coins_trending": [c["symbol"].upper() for c in our_coins_trending],
        "market_attention": "high" if len(our_coins_trending) > 0 else "normal",
    }

    log.info(f"CoinGecko trending: {[c['symbol'] for c in coins[:5]]}")
    return result


def run():
    """Main execution — gather all sentiment data and aggregate."""
    log.info("=" * 50)
    log.info("SENTIMENT AGENT — Starting collection")
    log.info("=" * 50)

    existing = load_data("sentiment.json")
    history = existing.get("history", [])

    # Collect from all sources
    fgi = fetch_fear_greed()
    news = fetch_cryptopanic()
    reddit = fetch_reddit_sentiment()
    trending = fetch_coingecko_trending()

    # Build aggregate sentiment signal
    signals = []

    if fgi and fgi.get("signal"):
        signals.append(fgi["signal"])

    if news:
        overall = news.get("overall", {})
        ratio = overall.get("ratio", 0.5)
        if ratio > 0.55:
            signals.append({"bias": "bullish", "score": int((ratio - 0.5) * 200),
                            "weight": 1.0, "reason": f"News ratio {ratio:.2f}"})
        elif ratio < 0.45:
            signals.append({"bias": "bearish", "score": int((0.5 - ratio) * 200),
                            "weight": 1.0, "reason": f"News ratio {ratio:.2f}"})

    if reddit:
        ws = reddit.get("weighted_score", 0)
        if abs(ws) > 0.05:
            signals.append({
                "bias": "bullish" if ws > 0 else "bearish",
                "score": int(abs(ws) * 100),
                "weight": 0.8,
                "reason": f"Reddit weighted {ws:.3f}",
            })

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

    if agg_score > 25:
        agg_bias = "bullish"
    elif agg_score < -25:
        agg_bias = "bearish"
    else:
        agg_bias = "neutral"

    agg_strength = "strong" if abs(agg_score) > 60 else "moderate" if abs(agg_score) > 35 else "weak"

    now = datetime.now(timezone.utc)
    snapshot = {
        "timestamp": now.isoformat(),
        "ts": int(time.time()),
        "fgi": fgi.get("current") if fgi else None,
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
        "fear_greed": fgi,
        "news": news,
        "reddit": reddit,
        "trending": trending,
        "history": history,
    }

    save_data("sentiment.json", result)

    log.info(f"AGGREGATE: score={agg_score} bias={agg_bias} strength={agg_strength}")
    log.info("Sentiment agent complete.")
    return result


if __name__ == "__main__":
    run()
