# CryptoAgent v2.1 — Agent Instructions

**RE-READ THE ROOT `/CLAUDE.md` BEFORE EVERY TASK. Re-read THIS file before touching any CryptoAgent code.**

## What This App Does

Serverless crypto trading agent running every 10 minutes on GitHub Actions. Based on AdaptiveTrend algorithm (arXiv 2602.11708, Sharpe 2.41, 6H timeframe). 7 strategies + 5 intel sub-agents for sentiment/whales/liquidations/on-chain/news. Trades on Coinbase Advanced Trade.

## Architecture

| Component | Stack | Location | Deploy |
|-----------|-------|----------|--------|
| Main Agent | Python | `agent.py` | GitHub Actions (every 10 min) |
| Market Data | Python | `data.py` | Imported by agent |
| Exchange API | Python | `exchange.py` | Coinbase Advanced Trade |
| Strategies | Python | `strategies.py` | 7 strategies |
| Indicators | Python | `indicators.py` | TA-Lib + TradingView |
| Intel Agents | Python | `intel/` | 5 sub-agents on GitHub Actions |
| State | JSON | `state.json` | Committed by Actions workflow |

## Backend Tiers Used

- **Tier 1 (REST API):** Coinbase Advanced Trade API (free), CoinGecko (free), Fear & Greed Index (free), CryptoPanic (free tier)
- **Tier 3 (Serverless Worker):** GitHub Actions cron every 10 minutes — 100% serverless, zero cost (public repo)

## 7 Trading Strategies

1. **Adaptive Trend** — Primary. 6H candles, daily trend filter. Paper's best performer.
2. **Momentum Squeeze** (BB-KC) — Bollinger Bands squeeze against Keltner Channels. Fewer but bigger trades.
3. **Funding Rate Fade** — Counter-trade extreme funding rates.
4. **Smart Money Divergence** — Volume/price divergence detection.
5. **CVD Divergence** — Cumulative Volume Delta divergence.
6. **Liquidation Cascade** — Trade liquidation level clusters.
7. **Multi-TF Confluence** — Multi-timeframe agreement signals.

## 5 Intel Sub-Agents

| Agent | Sources | File |
|-------|---------|------|
| Sentiment | Fear & Greed, Reddit, CryptoPanic | `intel/sentiment_agent.py` |
| Whales | Large transaction tracking | `intel/whale_agent.py` |
| Liquidations | OKX, Bybit cascade levels | `intel/liquidation_agent.py` |
| On-Chain | DeFi Llama, CoinGecko, CMC | `intel/onchain_agent.py` |
| News | RSS feeds, CryptoPanic | `intel/news_agent.py` |

## Key Config (config.py)

```python
MIN_CONFIDENCE = 0.65        # Minimum signal confidence to trade
MIN_EXPECTED_MOVE = 2%       # Must clear 0.80% round-trip fee drag
TIMEFRAME = "6H"             # Primary: 6-hour candles (not 1H)
TRAILING_STOP = 2.5          # ATR multiplier (paper's optimal)
ORDER_TYPE = "post_only"     # Guaranteed 0.40% maker fee
```

## Coinbase

- Account: mushman0011@proton.me
- Portfolio: ~$475
- API: Coinbase Advanced Trade (JWT auth)
- Fees: 0.40% maker (post_only limit orders)

## GitHub Actions Secrets

| Secret | Purpose |
|--------|---------|
| `COINBASE_KEY_JSON` | Coinbase API credentials |
| `CRYPTOPANIC_API_KEY` | CryptoPanic news API |
| `GROQ_API_KEY` | Groq LLM for intel analysis |

## Deploy

This runs 100% on GitHub Actions. No deploy needed beyond pushing code:
```bash
git push  # GitHub Actions picks up changes automatically
```

State is persisted in `state.json` — committed back by the workflow after each run.

## Dual-Push

After pushing to this monorepo, sync to origin:
```bash
../../scripts/sync-to-origin.sh crypto-agent
```
**Origin:** `jacksondurand95-debug/CryptoAgent-v2` (public)

## Critical Rules

- NEVER change MIN_CONFIDENCE below 0.65 or MIN_EXPECTED_MOVE below 2%
- ALWAYS use post_only limit orders (maker fees only)
- Primary timeframe is 6H — do not switch to 1H
- Trailing stop at 2.5x ATR — this is the paper's optimal, don't change without research
- state.json must be committed by the workflow after every run
- Zero cost — all APIs free tier, public repo = unlimited Actions minutes
