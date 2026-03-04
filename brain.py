"""CryptoAgent v3.0 — Beast Mode Dual-Brain Trading Engine.

Primary: Claude Sonnet (Anthropic) — best reasoning for complex market analysis
Fallback: Grok 4 (xAI) — unrestricted, fast, no safety theater

Multi-exchange derivatives intel. Coinbase One reduced fees.
If Claude API key has credits, it runs Claude. If not, Grok takes over.
Both get the same market data and system prompt.
"""
import json
import logging
import os
import time

log = logging.getLogger("brain")

# API endpoints
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# Models
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6-20250514")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4-0709")

SYSTEM_PROMPT = """You are an autonomous crypto trading algorithm managing a LIVE Coinbase portfolio. You are NOT an advisor. You ARE the decision engine. Your job is to MAKE MONEY. Output decisions, not opinions.

## ACCOUNT STATUS
- Coinbase One subscriber — 25% fee rebate on Advanced trades
- Effective round-trip: ~0.90% with limit orders (post_only maker)
- This means ANY move > 1.2% is profitable after fees. TRADE MORE.
- ~$480 account. Small but growing. Compound gains.

## HARD RULES
- Max position: 30% of portfolio. Max 3 concurrent positions.
- ALWAYS set stop losses. Default: 2x ATR from entry.
- NEVER widen a stop. NEVER average down.
- Use limit orders (post_only) for entries. Market orders only for emergency exits.

## YOUR INTELLIGENCE ADVANTAGE
You receive cross-exchange institutional data every 10 minutes from 3+ exchanges:

**Technical Analysis (multi-timeframe):**
- 6H (primary signal): RSI, MACD, EMA 9/21/50, BB, KC, ATR, ADX, StochRSI, momentum 7/14/28, rolling Sharpe, VWAP
- 1D (trend filter): Same indicators for trend confirmation
- 1H (fast): Quick momentum reads, entry timing

**Derivatives Intelligence (Bybit + Binance + OKX):**
- Funding rates across 3 exchanges (aggregate + individual)
- Open interest changes (rising/falling across exchanges)
- Long/short ratios (global + top trader positioning)
- Taker buy/sell volume ratios (who's aggressive)
- Smart money vs retail divergence

**Order Book:**
- Bid/ask depth imbalance on Coinbase (buy pressure vs sell pressure)
- Spread analysis

**Sentiment & News:**
- Fear & Greed Index (0-100)
- CryptoPanic hot news with sentiment scores
- 5 intel sub-agents: sentiment, whales, liquidations, on-chain, news

## DECISION FRAMEWORK — BE AGGRESSIVE

### Step 1: REGIME IDENTIFICATION (this determines everything)
- **TRENDING** — ADX > 25, EMAs aligned, momentum strong → RIDE IT. Trail stops.
- **RANGING** — ADX < 20, price bouncing between BB bands → Mean revert at extremes.
- **VOLATILE** — ATR expanding, funding extreme, liquidations happening → HIGHEST ALPHA. Trade the dislocations.
- **QUIET** — Low ADX, tight BBs, low volume → Only enter if squeeze is building (BB inside KC).

### Step 2: FIND THE EDGE (ranked by historical alpha)
1. **Liquidation cascade reversal** — Extreme fear + funding flush + OI collapse + price recovering = BUY THE BLOOD. This is the single highest-alpha pattern in crypto. Don't miss it.
2. **Funding rate fade** — All 3 exchanges showing extreme funding? The market is about to reverse. Negative FR = short squeeze incoming. Extreme positive = dump incoming.
3. **Smart money divergence** — Top traders going long while retail is short? FOLLOW THE WHALES. The divergence signal from Binance top trader ratio is worth 2x any technical signal.
4. **BB-KC squeeze breakout** — Bollinger Bands inside Keltner Channels for 3+ bars, then releasing. Direction from momentum. High conviction breakout.
5. **Order book imbalance** — Heavy bid depth (>15% imbalance) = buy pressure building. Heavy ask depth = sell pressure. This is REAL-TIME supply/demand.
6. **Multi-TF momentum** — 6H and 1D both trending same direction with ADX > 25. High probability continuation.
7. **Taker flow divergence** — Price dropping but taker buy ratio > 1.1 = hidden accumulation. Price rising but taker sell ratio > 1.1 = distribution.
8. **CVD divergence** — Price making new highs but cumulative volume delta diverging = distribution top. Vice versa for accumulation bottom.

### Step 3: CROSS-VALIDATE WITH DERIVATIVES
Required confluence for HIGH CONFIDENCE trades:
- At least 2 exchanges agreeing on funding direction
- Smart money positioning aligning with technical signal
- Taker flow confirming (not diverging from) the trade direction
- OI context: rising OI on breakout = real, falling OI on breakout = fake

### Step 4: EXECUTE DECISIVELY
When the signal is there, TAKE IT. Don't wait for perfect entries. The edge decays every 10 minutes.

## POSITION MANAGEMENT — ACTIVE, NOT PASSIVE
Review EVERY open position EVERY cycle. Recommend closing if:
- Original thesis is DEAD (funding flipped, momentum reversed, regime changed)
- Price is chopping sideways and capital is locked (opportunity cost)
- Better setup exists but capital is tied up
- Time stop: held > 72 hours without meaningful move
- Unrealized loss > 3% with no catalyst for recovery

CUT LOSERS FAST. LET WINNERS RIDE. This is the only rule that matters.

## LEARNING — YOU HAVE MEMORY
Look at recent trade history. If:
- All recent trades lost → You're fighting the trend. Reverse bias or sit out.
- Wins but small → Tighten entries, widen targets. Let winners run.
- Losses from fees → Move threshold was too low. Raise it.
- Stopped out then price reversed → Stops too tight. Widen slightly.
Adapt. Don't repeat mistakes.

## 6 TRADABLE PAIRS
BTC-USD, ETH-USD, SOL-USD, DOGE-USD, AVAX-USD, LINK-USD
Pick the BEST setup across ALL pairs. Quality over quantity.

## OUTPUT — RAW JSON ONLY
No markdown. No code fences. No commentary. No disclaimers. Just JSON.
{
  "action": "buy" | "sell" | "hold",
  "pair": "BTC-USD" | "ETH-USD" | "SOL-USD" | "DOGE-USD" | "AVAX-USD" | "LINK-USD" | null,
  "confidence": 0.0-1.0,
  "size_pct": 10-30,
  "entry_price": float | null,
  "stop_loss": float | null,
  "take_profit": float | null,
  "atr": float | null,
  "strategy": "strategy_name",
  "reasoning": "2-3 sentences. Be SPECIFIC. Name the data points driving this.",
  "market_regime": "trending" | "ranging" | "volatile" | "quiet",
  "position_review": [{"pair": "X-USD", "action": "hold"|"close", "reason": "why"}],
  "conviction_factors": ["factor1", "factor2", "factor3"]
}

Rules:
- "hold" → pair/entry/stop/tp = null
- ONE signal only — the single best play across all 6 pairs
- If nothing clears the bar, HOLD. But don't hold forever — this market moves.
- When you DO trade, go BIG (20-30% sizing) with TIGHT risk (2x ATR stop)
- Include conviction_factors — the 3 most important data points driving your decision"""


def _build_market_context(all_pair_data, state, portfolio_value, intel_brief, fgi):
    """Build compact market context for the LLM."""
    lines = []
    lines.append(f"PORTFOLIO: ${portfolio_value:,.2f} (started ${state.get('starting_value', portfolio_value):,.2f})")
    pnl = portfolio_value - state.get('starting_value', portfolio_value)
    pnl_pct = (pnl / state.get('starting_value', portfolio_value)) * 100 if state.get('starting_value') else 0
    lines.append(f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")

    # Actual fee rate if detected
    actual_fees = state.get("detected_fees")
    if actual_fees:
        lines.append(f"FEES: maker={actual_fees.get('maker', 'N/A')} taker={actual_fees.get('taker', 'N/A')} (Coinbase One rebate applied)")

    # Open positions with current unrealized P&L
    positions = state.get("positions", [])
    if positions:
        lines.append(f"\nOPEN POSITIONS ({len(positions)}):")
        for pos in positions:
            held_hrs = (time.time() - pos.get("opened_at", time.time())) / 3600
            curr_price_data = all_pair_data.get(pos['pair'], {}).get('ind_6h', {})
            curr_price = curr_price_data.get('price', pos['entry_price'])
            unrealized_pct = ((curr_price - pos['entry_price']) / pos['entry_price']) * 100
            unrealized_usd = pos.get('qty', 0) * (curr_price - pos['entry_price'])
            lines.append(
                f"  {pos['pair']}: {pos['side']} @ ${pos['entry_price']:,.2f} | "
                f"now=${curr_price:,.2f} ({unrealized_pct:+.1f}% / ${unrealized_usd:+.2f}) | "
                f"qty={pos.get('qty', 0):.6f} (${pos.get('usd_amount', 0):.2f}) | "
                f"SL=${pos.get('stop_loss', 0):,.2f} TP=${pos.get('take_profit', 0):,.2f} | "
                f"held={held_hrs:.1f}h | peak=${pos.get('highest_price', 0):,.2f}"
            )
    else:
        lines.append("\nNO OPEN POSITIONS — CAPITAL AVAILABLE")

    # Pending orders
    pending = state.get("pending_orders", [])
    if pending:
        lines.append(f"\nPENDING ORDERS: {len(pending)}")
        for o in pending:
            age = (time.time() - o.get("placed_at", time.time())) / 60
            lines.append(f"  {o['side']} {o['pair']} @ ${o.get('price', 0):,.2f} ({age:.0f}min old)")

    # Recent trades — the bot should learn from these
    trades = state.get("trades", [])[-15:]
    if trades:
        total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
        total_fees = sum(t.get("fees_usd", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        avg_hold = sum(t.get("duration_min", 0) for t in trades) / len(trades)
        lines.append(f"\nLAST {len(trades)} TRADES: {wins}W/{len(trades)-wins}L | P&L: ${total_pnl:+,.2f} | Fees: ${total_fees:.2f} | Avg Hold: {avg_hold:.0f}min")
        for t in trades[-5:]:  # Show last 5 in detail
            lines.append(
                f"  {t['pair']} {t['side']} ${t['entry_price']:,.2f}->${t['exit_price']:,.2f} "
                f"P&L=${t['pnl_usd']:+,.2f} fees=${t.get('fees_usd', 0):.2f} "
                f"[{t.get('reason', '?')}] held={t.get('duration_min', 0):.0f}min"
            )
        if total_pnl < 0:
            lines.append(f"  *** NET NEGATIVE (${total_pnl:+,.2f}). Fees={total_fees:.2f}. ADAPT. ***")
        if wins == 0 and len(trades) >= 3:
            lines.append(f"  *** {len(trades)} CONSECUTIVE LOSSES. Consider reversing bias or sitting out. ***")

    # Per-pair market data
    for pair, data in all_pair_data.items():
        ind = data.get("ind_6h", {})
        ind_d = data.get("ind_1d", {})
        ind_1h = data.get("ind_1h", {})
        deriv = data.get("derivatives", {})
        book = data.get("orderbook", {})
        tv = data.get("tv", {})

        lines.append(f"\n{'='*60}")
        lines.append(f"{pair} — ${ind.get('price', 0):,.2f}")
        lines.append(f"{'='*60}")

        # 6H technicals
        lines.append(f"6H: RSI={ind.get('rsi', '?')} MACD_hist={ind.get('macd_hist', '?')} "
                     f"ADX={ind.get('adx', '?')} DI+={ind.get('di_plus', '?')} DI-={ind.get('di_minus', '?')}")
        lines.append(f"    EMA: 9={ind.get('ema_9', '?')} 21={ind.get('ema_21', '?')} 50={ind.get('ema_50', '?')} trend={ind.get('ema_trend', '?')}")
        lines.append(f"    BB: {ind.get('bb_lower', '?')}/{ind.get('bb_mid', '?')}/{ind.get('bb_upper', '?')} pct={ind.get('bb_pct', '?')}")
        lines.append(f"    KC: {ind.get('kc_lower', '?')}/{ind.get('kc_upper', '?')} squeeze={ind.get('squeeze', False)} bars={ind.get('squeeze_bars', 0)} releasing={ind.get('squeeze_releasing', False)}")
        lines.append(f"    ATR={ind.get('atr', '?')} ({ind.get('atr_pct', '?')}%) vol={ind.get('vol_ratio', '?')}x")
        lines.append(f"    StochRSI: K={ind.get('stoch_k', '?')} D={ind.get('stoch_d', '?')}")
        lines.append(f"    Momentum: 7d={ind.get('momentum_7', '?')} 14d={ind.get('momentum_14', '?')} 28d={ind.get('momentum_28', '?')}")
        lines.append(f"    Sharpe={ind.get('rolling_sharpe', '?')} VWAP={ind.get('vwap', '?')}")
        lines.append(f"    24h: {ind.get('24h_change_pct', '?')}% high={ind.get('high_24h', '?')} low={ind.get('low_24h', '?')}")

        # 1D trend
        if ind_d:
            lines.append(f"1D: trend={ind_d.get('ema_trend', '?')} RSI={ind_d.get('rsi', '?')} "
                         f"ADX={ind_d.get('adx', '?')} mom14={ind_d.get('momentum_14', '?')} "
                         f"mom28={ind_d.get('momentum_28', '?')}")

        # 1H fast
        if ind_1h:
            lines.append(f"1H: RSI={ind_1h.get('rsi', '?')} MACD_hist={ind_1h.get('macd_hist', '?')} "
                         f"mom7={ind_1h.get('momentum_7', '?')} squeeze={ind_1h.get('squeeze', False)}")

        # Cross-exchange derivatives
        if deriv:
            agg = deriv.get("aggregate", {})
            feed_count = deriv.get("feed_count", 0)
            lines.append(f"DERIVATIVES ({feed_count} feeds):")
            lines.append(f"    Aggregate: bias={agg.get('overall_bias', '?')} "
                         f"funding={agg.get('funding_bias', '?')} "
                         f"smart_money={agg.get('smart_money_signal', '?')} "
                         f"taker={agg.get('taker_flow', '?')} "
                         f"positioning={agg.get('positioning_bias', '?')}")
            lines.append(f"    Signals: {agg.get('bullish_signals', 0)} bullish / {agg.get('bearish_signals', 0)} bearish")

            # Individual exchange details
            for key in ("bybit_funding", "binance_funding", "okx_funding"):
                f = deriv.get(key)
                if f:
                    lines.append(f"    {f['exchange']} FR: {f['current']:.4f}%")

            bybit_ticker = deriv.get("bybit_ticker")
            if bybit_ticker:
                lines.append(f"    Bybit: vol24h={bybit_ticker.get('volume_24h', 0):,.0f} "
                             f"OI={bybit_ticker.get('open_interest', 0):,.0f} "
                             f"24h={bybit_ticker.get('price_change_24h_pct', 0):+.1f}%")

            top = deriv.get("binance_top_traders")
            if top:
                lines.append(f"    Top Traders: ratio={top.get('top_trader_ratio', '?')} "
                             f"whalesLong={top.get('whales_long', False)} whalesShort={top.get('whales_short', False)}")

            taker = deriv.get("binance_taker")
            if taker:
                lines.append(f"    Taker: ratio={taker.get('ratio', '?')} "
                             f"aggressiveBuyers={taker.get('aggressive_buyers', False)} "
                             f"aggressiveSellers={taker.get('aggressive_sellers', False)}")

        # Order book
        if book:
            lines.append(f"BOOK: bid_depth={book.get('bid_depth', '?')} ask_depth={book.get('ask_depth', '?')} "
                         f"imbalance={book.get('imbalance', '?')} signal={book.get('imbalance_signal', '?')} "
                         f"spread={book.get('spread_pct', '?')}%")

        # TradingView
        if tv:
            lines.append(f"TV: {tv.get('RECOMMENDATION', '?')} ({tv.get('BUY', 0)}B/{tv.get('SELL', 0)}S/{tv.get('NEUTRAL', 0)}N)")

    # Intel sub-agents
    if intel_brief:
        agg = intel_brief.get("aggregate", {})
        lines.append(f"\nINTEL: score={agg.get('score', 0)} bias={agg.get('bias', '?')} "
                     f"str={agg.get('strength', '?')} conf={agg.get('confidence', 0):.0%} "
                     f"({agg.get('available_sources', 0)}/{agg.get('total_sources', 0)} sources)")
        coins = intel_brief.get("coins", {})
        for coin, cd in coins.items():
            lines.append(f"  {coin}: {cd.get('bias', '?')} score={cd.get('score', '?')}")
        for evt in intel_brief.get("alpha_events", [])[:5]:
            lines.append(f"  ALPHA [{evt.get('importance', '?')}]: {evt.get('title', '')[:100]}")

    # News
    news = all_pair_data.get("_news", [])
    if news:
        lines.append(f"\nNEWS ({len(news)} items):")
        for n in news[:5]:
            sentiment = "+" if n.get("sentiment", 0) > 0 else "-" if n.get("sentiment", 0) < 0 else "~"
            coins = ",".join(n.get("currencies", []))
            lines.append(f"  [{sentiment}] [{coins}] {n.get('title', '')[:80]}")

    if fgi:
        lines.append(f"\nFEAR&GREED: {fgi['value']} ({fgi['classification']})")

    lines.append(f"\nTIME: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    lines.append("DECISION:")

    return "\n".join(lines)


def _call_claude(api_key, context):
    """Call Claude API (Anthropic native format)."""
    import requests

    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 1500,
            "temperature": 0.2,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": context}],
        },
        timeout=90,
    )

    if resp.status_code != 200:
        error_msg = resp.text[:200]
        log.warning(f"Claude API error {resp.status_code}: {error_msg}")
        return None, f"claude_error_{resp.status_code}"

    data = resp.json()
    content = data.get("content", [{}])[0].get("text", "")
    usage = data.get("usage", {})
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    return content, f"claude [{CLAUDE_MODEL}] ({tokens} tok)"


def _call_grok(api_key, context):
    """Call Grok API (OpenAI-compatible format)."""
    import requests

    resp = requests.post(
        XAI_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": XAI_MODEL,
            "max_tokens": 1500,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
        },
        timeout=120,
    )

    if resp.status_code != 200:
        error_msg = resp.text[:200]
        log.warning(f"Grok API error {resp.status_code}: {error_msg}")
        return None, f"grok_error_{resp.status_code}"

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)

    return content, f"grok [{XAI_MODEL}] ({tokens} tok)"


def _parse_response(content):
    """Parse LLM JSON response, stripping code fences if present."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    if content.startswith("json"):
        content = content[4:].strip()
    return json.loads(content)


def analyze(all_pair_data, state, portfolio_value, intel_brief=None, fgi=None):
    """Analyze market data using Claude (primary) or Grok (fallback).

    Returns:
        Signal dict compatible with agent.py or None
    """
    claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
    grok_key = os.environ.get("XAI_API_KEY", "")

    if not claude_key and not grok_key:
        log.error("No API keys set (need ANTHROPIC_API_KEY or XAI_API_KEY)")
        return None

    context = _build_market_context(all_pair_data, state, portfolio_value, intel_brief, fgi)

    content = None
    brain_id = None

    # Try Claude first
    if claude_key:
        try:
            content, brain_id = _call_claude(claude_key, context)
            if content:
                log.info(f"BRAIN: Using {brain_id}")
        except Exception as e:
            log.warning(f"Claude failed: {e}")
            content = None

    # Fallback to Grok
    if not content and grok_key:
        try:
            content, brain_id = _call_grok(grok_key, context)
            if content:
                log.info(f"BRAIN: Fallback to {brain_id}")
        except Exception as e:
            log.error(f"Grok also failed: {e}")
            return None

    if not content:
        log.error("Both Claude and Grok failed — no brain available")
        return None

    # Parse decision
    try:
        decision = _parse_response(content)
    except json.JSONDecodeError as e:
        log.error(f"Brain returned invalid JSON: {e}")
        log.error(f"  Raw: {content[:300]}")
        return None

    # Log decision
    log.info(f"DECISION [{brain_id}]: action={decision.get('action')} "
             f"pair={decision.get('pair')} conf={decision.get('confidence', 0):.2f} "
             f"regime={decision.get('market_regime')} strategy={decision.get('strategy')}")
    log.info(f"  >> {decision.get('reasoning', 'N/A')[:200]}")
    log.info(f"  Conviction: {decision.get('conviction_factors', [])}")

    for review in decision.get("position_review", []):
        log.info(f"  REVIEW {review['pair']}: {review['action']} — {review.get('reason', '')[:100]}")

    # Validate
    action = decision.get("action", "hold")
    if action == "hold":
        log.info("  => HOLD")
        # Still return position reviews even on hold
        if decision.get("position_review"):
            return {
                "action": "hold",
                "pair": None,
                "_position_reviews": decision.get("position_review", []),
            }
        return None

    confidence = decision.get("confidence", 0)
    min_conf = float(os.environ.get("MIN_CONFIDENCE", "0.55"))
    if confidence < min_conf:
        log.info(f"  => Confidence {confidence:.2f} < {min_conf} — forcing HOLD")
        return None

    # Determine brain tag
    brain_tag = "claude" if "claude" in (brain_id or "") else "grok"

    signal = {
        "action": action,
        "pair": decision.get("pair"),
        "confidence": confidence,
        "size_pct": decision.get("size_pct", 25),
        "entry_price": decision.get("entry_price"),
        "stop_loss": decision.get("stop_loss"),
        "take_profit": decision.get("take_profit"),
        "atr": decision.get("atr"),
        "strategy": f"{brain_tag}:{decision.get('strategy', 'unknown')}",
        "reasoning": decision.get("reasoning", f"{brain_tag} decision"),
        "conviction_factors": decision.get("conviction_factors", []),
        "_position_reviews": decision.get("position_review", []),
    }

    return signal
