#!/usr/bin/env python3
"""Intel Sub-Agent: News & Alpha Scraper.

Sources (all free, no API keys required):
1. CoinDesk RSS feed — free
2. CoinTelegraph RSS feed — free
3. The Block RSS — free
4. Decrypt RSS — free
5. Bitcoin Magazine RSS — free
6. CryptoPanic trending posts — free
7. Google News RSS for crypto keywords — free

Runs: Every 2 hours via GitHub Actions
Output: intel/data/news.json
"""
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import retry_get, load_data, save_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("news")

# RSS feeds — all free, no auth. SCRUB EVERYTHING.
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/feed",
    "The Defiant": "https://thedefiant.io/feed",
    "DL News": "https://www.dlnews.com/arc/outboundfeeds/rss/",
    "Blockworks": "https://blockworks.co/feed",
    "The Block": "https://www.theblock.co/rss.xml",
    "Crypto Briefing": "https://cryptobriefing.com/feed/",
    "U.Today": "https://u.today/rss",
    "NewsBTC": "https://www.newsbtc.com/feed/",
    "BeInCrypto": "https://beincrypto.com/feed/",
    # Macro feeds that move crypto
    "Fed News (Google)": "https://news.google.com/rss/search?q=federal+reserve+interest+rate&hl=en-US",
    "Crypto Regulation (Google)": "https://news.google.com/rss/search?q=crypto+regulation+SEC+2026&hl=en-US",
    "Bitcoin ETF (Google)": "https://news.google.com/rss/search?q=bitcoin+ETF+inflow+outflow&hl=en-US",
}

# Keywords that affect crypto prices
MAJOR_BULLISH_KEYWORDS = {
    "etf approved": 90, "etf approval": 90, "institutional adoption": 70,
    "blackrock": 60, "fidelity": 60, "spot etf": 80,
    "fed rate cut": 70, "rate cut": 60, "dovish": 50, "pause rate": 50,
    "stablecoin bill": 50, "crypto friendly": 50, "pro crypto": 50,
    "bitcoin reserve": 80, "strategic reserve": 80,
    "mass adoption": 60, "billion investment": 70, "billion buy": 70,
    "all time high": 50, "ath": 40, "breakout": 40,
    "partnership": 30, "integration": 30,
    # Bottom signals — these fire when market is capitulating (BULLISH contrarian)
    "capitulation": 40, "bottom signal": 50, "oversold": 40,
    "accumulation zone": 50, "smart money buying": 60,
    "etf inflow": 60, "record inflow": 70, "net inflow": 50,
    "whale accumulation": 60, "microstrategy buy": 50, "saylor": 40,
    "treasury buy": 70, "sovereign buy": 80,
    "hash rate ath": 40, "miner capitulation over": 50,
    "short squeeze": 60, "liquidation cascade": 50,
}

MAJOR_BEARISH_KEYWORDS = {
    "sec lawsuit": -80, "sec sues": -80, "enforcement action": -70,
    "ban crypto": -90, "crypto ban": -90, "regulation": -30,
    "hack": -60, "exploit": -60, "rug pull": -70,
    "insolvency": -80, "bankruptcy": -80, "collapse": -70,
    "rate hike": -60, "hawkish": -50, "tightening": -40,
    "investigation": -40, "subpoena": -50, "fraud": -60,
    "liquidation": -40, "contagion": -70, "crash": -50,
    "sell off": -40, "dump": -30, "capitulation": -40,
}

# Coin mentions
COIN_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "satoshi"],
    "ETH": ["ethereum", "eth", "vitalik"],
    "SOL": ["solana", "sol"],
}


def strip_html(text):
    """Remove HTML tags from text."""
    clean = re.sub(r'<[^>]+>', '', text)
    return unescape(clean).strip()


def parse_rss(source_name, url):
    """Parse an RSS feed and extract articles."""
    r = retry_get(url, timeout=15)
    if not r:
        log.warning(f"Failed to fetch {source_name} RSS")
        return []

    articles = []
    try:
        root = ET.fromstring(r.content)

        # Handle both RSS 2.0 and Atom
        # RSS 2.0: channel/item
        items = root.findall('.//item')
        if not items:
            # Atom: entry
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            items = root.findall('.//atom:entry', ns)

        for item in items[:20]:
            title = ""
            description = ""
            link = ""
            pub_date = ""

            # RSS 2.0
            title_el = item.find('title')
            if title_el is not None and title_el.text:
                title = title_el.text.strip()

            desc_el = item.find('description')
            if desc_el is not None and desc_el.text:
                description = strip_html(desc_el.text)[:500]

            link_el = item.find('link')
            if link_el is not None:
                link = (link_el.text or link_el.get('href', '')).strip()

            date_el = item.find('pubDate')
            if date_el is not None and date_el.text:
                pub_date = date_el.text.strip()

            if not title:
                continue

            articles.append({
                "source": source_name,
                "title": title,
                "description": description[:300],
                "link": link,
                "published": pub_date,
            })

    except ET.ParseError as e:
        log.warning(f"XML parse error for {source_name}: {e}")
    except Exception as e:
        log.warning(f"Error parsing {source_name}: {e}")

    return articles


def analyze_article(article):
    """Analyze a single article for sentiment and relevance."""
    title = article.get("title", "").lower()
    desc = article.get("description", "").lower()
    combined = f"{title} {desc}"

    # Score based on keyword matches
    score = 0
    matched_keywords = []

    for keyword, weight in MAJOR_BULLISH_KEYWORDS.items():
        if keyword in combined:
            score += weight
            matched_keywords.append(f"+{keyword}")

    for keyword, weight in MAJOR_BEARISH_KEYWORDS.items():
        if keyword in combined:
            score += weight  # weight is already negative
            matched_keywords.append(f"{keyword}")

    # Determine which coins are mentioned
    coins_mentioned = []
    for coin, keywords in COIN_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            coins_mentioned.append(coin)

    # Classify
    if score > 30:
        sentiment = "bullish"
    elif score < -30:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    # Importance (absolute score)
    abs_score = abs(score)
    if abs_score > 70:
        importance = "critical"
    elif abs_score > 40:
        importance = "high"
    elif abs_score > 15:
        importance = "medium"
    else:
        importance = "low"

    article["sentiment"] = sentiment
    article["score"] = score
    article["importance"] = importance
    article["coins"] = coins_mentioned
    article["keywords"] = matched_keywords

    return article


def fetch_google_news_crypto():
    """Fetch crypto-related news from Google News RSS (free, no auth)."""
    log.info("Fetching Google News crypto headlines...")
    queries = [
        "bitcoin+crypto+market",
        "ethereum+defi+regulation",
        "crypto+SEC+regulation",
    ]

    articles = []
    for query in queries:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        r = retry_get(url, timeout=15)
        if not r:
            continue

        try:
            root = ET.fromstring(r.content)
            items = root.findall('.//item')
            for item in items[:10]:
                title_el = item.find('title')
                link_el = item.find('link')
                date_el = item.find('pubDate')

                if title_el is not None and title_el.text:
                    articles.append({
                        "source": "Google News",
                        "title": title_el.text.strip(),
                        "description": "",
                        "link": link_el.text.strip() if link_el is not None and link_el.text else "",
                        "published": date_el.text.strip() if date_el is not None and date_el.text else "",
                    })
        except Exception as e:
            log.warning(f"Google News parse error: {e}")

        time.sleep(1)

    return articles


def detect_alpha_events(articles):
    """Detect high-impact events from articles that could move markets."""
    alpha_events = []

    for article in articles:
        if article.get("importance") in ("critical", "high"):
            alpha_events.append({
                "title": article["title"],
                "source": article["source"],
                "sentiment": article["sentiment"],
                "score": article["score"],
                "importance": article["importance"],
                "coins": article.get("coins", []),
                "keywords": article.get("keywords", []),
            })

    # Sort by absolute score (importance)
    alpha_events.sort(key=lambda x: abs(x["score"]), reverse=True)
    return alpha_events[:10]


def fetch_x_sentiment_via_grok(api_key):
    """Use Grok (which has live X/Twitter access) to get crypto Twitter sentiment.

    Grok sees real-time tweets. We ask it to summarize what crypto Twitter is saying.
    Returns list of synthetic articles from X sentiment.
    """
    import requests as req
    try:
        resp = req.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.environ.get("XAI_MODEL", "grok-3-mini-fast"),
                "max_tokens": 800,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "You are a crypto market analyst monitoring X/Twitter in real-time. Output JSON only."},
                    {"role": "user", "content": """Scan crypto Twitter right now. What are the top 5-8 trending narratives/events being discussed?

For each, output: title (headline style), sentiment (bullish/bearish/neutral), coins mentioned, importance (critical/high/medium/low).

Focus on: whale moves, ETF flows, regulatory news, exchange issues, macro events, liquidation cascades, notable trader calls.

JSON format:
{"tweets": [{"title": "...", "sentiment": "bullish", "coins": ["BTC"], "importance": "high"}]}"""},
                ],
            },
            timeout=30,
        )
        if resp.status_code != 200:
            log.warning(f"Grok X scrape failed: {resp.status_code}")
            return []

        content = resp.json()["choices"][0]["message"]["content"]
        # Parse JSON from response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        if content.startswith("json"):
            content = content[4:]

        data = json.loads(content.strip())
        articles = []
        for tweet in data.get("tweets", []):
            articles.append({
                "title": tweet.get("title", ""),
                "description": f"X/Twitter: {tweet.get('title', '')}",
                "source": "X/Twitter (Grok)",
                "link": "",
                "published": "",
                "coins_mentioned": tweet.get("coins", []),
                "_importance": tweet.get("importance", "medium"),
                "_sentiment": tweet.get("sentiment", "neutral"),
            })
        log.info(f"X/Twitter: {len(articles)} narratives from Grok")
        return articles
    except Exception as e:
        log.warning(f"Grok X scrape error: {e}")
        return []


def run():
    """Main execution."""
    log.info("=" * 50)
    log.info("NEWS AGENT — Starting collection")
    log.info("=" * 50)

    existing = load_data("news.json")
    history = existing.get("history", [])

    # Fetch from all RSS feeds
    all_articles = []

    for source_name, url in RSS_FEEDS.items():
        log.info(f"Fetching {source_name}...")
        articles = parse_rss(source_name, url)
        all_articles.extend(articles)
        time.sleep(1)  # Rate limit between feeds

    # Google News
    google_articles = fetch_google_news_crypto()
    all_articles.extend(google_articles)

    # X/Twitter via Grok — real-time crypto CT sentiment
    xai_key = os.environ.get("XAI_API_KEY", "")
    if xai_key:
        x_articles = fetch_x_sentiment_via_grok(xai_key)
        all_articles.extend(x_articles)
    else:
        log.info("No XAI_API_KEY — skipping X/Twitter scrape")

    log.info(f"Total articles fetched: {len(all_articles)}")

    # Analyze each article
    analyzed = []
    for article in all_articles:
        analyzed.append(analyze_article(article))

    # Sort by score magnitude
    analyzed.sort(key=lambda x: abs(x.get("score", 0)), reverse=True)

    # Detect alpha events
    alpha = detect_alpha_events(analyzed)

    # Aggregate sentiment
    scored = [a for a in analyzed if a.get("score", 0) != 0]
    total_positive = sum(a["score"] for a in scored if a["score"] > 0)
    total_negative = sum(a["score"] for a in scored if a["score"] < 0)
    net_score = total_positive + total_negative

    # Per-coin sentiment
    coin_sentiment = {}
    for coin in COIN_KEYWORDS:
        coin_articles = [a for a in analyzed if coin in a.get("coins", [])]
        if coin_articles:
            coin_score = sum(a.get("score", 0) for a in coin_articles)
            coin_sentiment[coin] = {
                "articles": len(coin_articles),
                "score": coin_score,
                "bias": "bullish" if coin_score > 20 else "bearish" if coin_score < -20 else "neutral",
            }

    # Normalize aggregate to -100 to +100
    if scored:
        max_possible = max(abs(total_positive), abs(total_negative), 1)
        agg_score = int(net_score / max_possible * 50)  # Scale to +-50 range
    else:
        agg_score = 0
    agg_score = max(-100, min(100, agg_score))

    agg_bias = "bullish" if agg_score > 15 else "bearish" if agg_score < -15 else "neutral"
    agg_strength = "strong" if abs(agg_score) > 50 else "moderate" if abs(agg_score) > 25 else "weak"

    now = datetime.now(timezone.utc)
    snapshot = {
        "timestamp": now.isoformat(),
        "ts": int(time.time()),
        "score": agg_score,
        "bias": agg_bias,
        "alpha_count": len(alpha),
    }
    history.append(snapshot)

    result = {
        "aggregate": {
            "score": agg_score,
            "bias": agg_bias,
            "strength": agg_strength,
            "total_articles": len(all_articles),
            "scored_articles": len(scored),
            "positive_score": total_positive,
            "negative_score": total_negative,
        },
        "alpha_events": alpha,
        "coin_sentiment": coin_sentiment,
        "top_articles": [
            {
                "title": a["title"],
                "source": a["source"],
                "sentiment": a["sentiment"],
                "score": a["score"],
                "importance": a["importance"],
                "coins": a.get("coins", []),
            }
            for a in analyzed[:30]
        ],
        "sources": {
            source: len([a for a in analyzed if a["source"] == source])
            for source in set(a["source"] for a in analyzed)
        },
        "history": history,
    }

    save_data("news.json", result)

    log.info(f"Alpha events: {len(alpha)}")
    for evt in alpha[:3]:
        log.info(f"  [{evt['importance']}] {evt['sentiment'].upper()}: {evt['title'][:80]}")
    log.info(f"AGGREGATE: score={agg_score} bias={agg_bias} strength={agg_strength}")
    log.info("News agent complete.")
    return result


if __name__ == "__main__":
    run()
