"""CryptoAgent v2.2 — Dual-Brain Trading Engine.

Primary: Claude Sonnet (Anthropic) — best reasoning for complex market analysis
Fallback: Grok 4 (xAI) — unrestricted, fast, no safety theater

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

SYSTEM_PROMPT = """You are an autonomous crypto trading algorithm managing a LIVE Coinbase portfolio with REAL money. You are not an advisor. You ARE the trader. Output decisions, not opinions.

## HARD CONSTRAINTS
- Round-trip fee: 0.80% (0.40% maker, post_only limit orders). NEVER trade unless expected move > 2%.
- Max position: 25% of portfolio. Max 2 concurrent positions.
- ALWAYS set stop losses. Default: 2.5x ATR from entry. NEVER widen a stop.
- Never average down on a losing position.
- This is a ~$460 account. Every dollar matters. Don't piss it away on marginal setups.

## YOUR DATA ADVANTAGE
You receive institutional-grade data every 10 minutes:
- Multi-timeframe technicals: 6H (primary) + 1D (trend filter) — RSI, MACD, EMA (9/21/50), BB, KC, ATR, ADX, StochRSI, momentum (7/14/28), rolling Sharpe
- OKX derivatives: funding rates (current + next), open interest, global long/short ratio, top trader positioning, taker buy/sell volume
- Smart money vs retail positioning divergence
- Fear & Greed Index (0-100)
- 5 intel sub-agents: sentiment, whale tracking, liquidation monitoring, on-chain metrics, news
- TradingView 4H consensus

## DECISION FRAMEWORK
1. **Identify regime FIRST**: trending / ranging / volatile / quiet
   - Quiet = NO EDGE. Hold. Fees will bleed you in chop.
2. **Match strategy to regime**:
   - Trending: Adaptive Trend (6H mom28 > 2% + daily EMA alignment). Sharpe 2.41 backtested. Let winners run.
   - Ranging: Mean revert off BB/KC extremes. Tight stops, take profit at boundaries.
   - Volatile: Liquidation cascade reversals, funding rate fades, smart money divergence. These are the highest-alpha setups.
3. **Cross-validate with derivatives**:
   - Funding < -0.02% = short squeeze setup (bullish)
   - Funding > 0.05% = overleveraged longs (bearish)
   - Top traders diverging from retail by > 0.4 = follow smart money
   - Taker buy/sell > 1.10 with price down = accumulation (bullish)
   - OI crashing + volume spiking + price recovering = cascade reversal (high alpha)
4. **Confluence matters**: Multiple signals agreeing = higher confidence. Single indicator = lower confidence.

## EDGE SIGNALS (ranked by historical alpha)
1. Liquidation cascade reversal — buy the blood after forced selling exhausts
2. Funding rate fade — extreme FR predicts mean reversion
3. Smart money divergence — whales know more than retail, follow them
4. BB-KC squeeze breakout — compressed volatility explodes directionally
5. CVD divergence — price/volume disagreement reveals hidden accumulation/distribution
6. Adaptive trend momentum — 28-period momentum with Sharpe filter
7. Multi-TF confluence — 6H + 1D both screaming same direction

## POSITION MANAGEMENT
Review EVERY open position. Recommend closing if:
- Original thesis is dead (funding flipped, momentum reversed, regime changed)
- Chopping sideways burning time and capital (opportunity cost)
- Better setup elsewhere but capital is locked
Cut losers FAST. Let winners ride with trailing stops.

## LEARNING FROM PAST TRADES
Look at the recent trade history. If past trades lost money due to small moves and fees, RAISE your threshold. Don't repeat the same mistake. Only trade when the setup is CLEARLY worth the fee drag.

## OUTPUT
Return ONLY raw JSON. No markdown fences. No commentary. No disclaimers.
{
  "action": "buy" | "sell" | "hold",
  "pair": "BTC-USD" | "ETH-USD" | "SOL-USD" | null,
  "confidence": 0.0-1.0,
  "size_pct": 10-25,
  "entry_price": float | null,
  "stop_loss": float | null,
  "take_profit": float | null,
  "atr": float | null,
  "strategy": "strategy_name",
  "reasoning": "2-3 sentences. Be SPECIFIC about what data is driving this decision.",
  "market_regime": "trending" | "ranging" | "volatile" | "quiet",
  "position_review": [{"pair": "X-USD", "action": "hold"|"close", "reason": "why"}]
}

- "hold" → pair/entry/stop/tp = null
- Empty position_review if no open positions
- ONE signal only — the single best play across all 3 pairs
- If nothing clears the 2% expected move bar with high confidence, HOLD
- When you DO trade, be decisive: high confidence, clear reasoning, tight risk management"""


def _build_market_context(all_pair_data, state, portfolio_value, intel_brief, fgi):
    """Build compact market context for the LLM."""
    lines = []
    lines.append(f"PORTFOLIO: ${portfolio_value:,.2f} (started ${state.get('starting_value', portfolio_value):,.2f})")
    pnl = portfolio_value - state.get('starting_value', portfolio_value)
    lines.append(f"P&L: ${pnl:+,.2f}")

    # Open positions with current unrealized P&L
    positions = state.get("positions", [])
    if positions:
        lines.append("\nOPEN POSITIONS:")
        for pos in positions:
            held_hrs = (time.time() - pos.get("opened_at", time.time())) / 3600
            curr_price_data = all_pair_data.get(pos['pair'], {}).get('ind_6h', {})
            curr_price = curr_price_data.get('price', pos['entry_price'])
            unrealized_pct = ((curr_price - pos['entry_price']) / pos['entry_price']) * 100
            lines.append(
                f"  {pos['pair']}: {pos['side']} @ ${pos['entry_price']:,.2f} | "
                f"now=${curr_price:,.2f} ({unrealized_pct:+.1f}%) | "
                f"qty={pos.get('qty', 0):.6f} (${pos.get('usd_amount', 0):.2f}) | "
                f"SL=${pos.get('stop_loss', 0):,.2f} TP=${pos.get('take_profit', 0):,.2f} | "
                f"held={held_hrs:.1f}h | peak=${pos.get('highest_price', 0):,.2f}"
            )
    else:
        lines.append("\nNO OPEN POSITIONS")

    # Pending orders
    pending = state.get("pending_orders", [])
    if pending:
        lines.append(f"\nPENDING ORDERS: {len(pending)}")
        for o in pending:
            age = (time.time() - o.get("placed_at", time.time())) / 60
            lines.append(f"  {o['side']} {o['pair']} @ ${o.get('price', 0):,.2f} ({age:.0f}min old)")

    # Recent trades — the bot should learn from these
    trades = state.get("trades", [])[-10:]
    if trades:
        total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
        total_fees = sum(t.get("fees_usd", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        lines.append(f"\nLAST {len(trades)} TRADES: {wins}W/{len(trades)-wins}L | P&L: ${total_pnl:+,.2f} | Fees: ${total_fees:.2f}")
        for t in trades:
            lines.append(
                f"  {t['pair']} {t['side']} ${t['entry_price']:,.2f}->${t['exit_price']:,.2f} "
                f"P&L=${t['pnl_usd']:+,.2f} fees=${t.get('fees_usd', 0):.2f} "
                f"[{t.get('reason', '?')}] {t.get('closed_at_str', '')}"
            )
        if total_pnl < 0:
            lines.append(f"  ** WARNING: Recent trades are NET NEGATIVE. Fees ({total_fees:.2f}) ate profits. Be more selective. **")

    # Per-pair market data
    for pair, data in all_pair_data.items():
        ind = data.get("ind_6h", {})
        ind_d = data.get("ind_1d", {})
        okx = data.get("onchain", {})
        tv = data.get("tv", {})

        lines.append(f"\n{'='*50}")
        lines.append(f"{pair} — ${ind.get('price', 0):,.2f}")
        lines.append(f"{'='*50}")

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

        if ind_d:
            lines.append(f"1D: trend={ind_d.get('ema_trend', '?')} RSI={ind_d.get('rsi', '?')} "
                         f"ADX={ind_d.get('adx', '?')} mom14={ind_d.get('momentum_14', '?')} "
                         f"mom28={ind_d.get('momentum_28', '?')}")

        if okx:
            fr = okx.get("funding", {})
            if fr:
                lines.append(f"FR: {fr.get('current', '?')}% next={fr.get('next', '?')}% "
                             f"bias={fr.get('signal', {}).get('bias', '?')} "
                             f"str={fr.get('signal', {}).get('strength', '?')}")
            oi = okx.get("open_interest", {})
            if oi:
                lines.append(f"OI: {oi.get('current', '?')} contracts")
            ls = okx.get("long_short_ratio", {})
            if ls:
                lines.append(f"Global L/S: {ls.get('current', '?')} xShort={ls.get('extreme_short', False)} xLong={ls.get('extreme_long', False)}")
            top = okx.get("top_traders", {})
            if top:
                lines.append(f"Top Traders: {top.get('ratio', '?')} whalesLong={top.get('whales_long', False)} whalesShort={top.get('whales_short', False)}")
            tk = okx.get("taker_ratio", {})
            if tk:
                lines.append(f"Taker: ratio={tk.get('ratio', '?')} aggressiveBuyers={tk.get('aggressive_buyers', False)}")

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
            "max_tokens": 1024,
            "temperature": 0.2,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": context}],
        },
        timeout=60,
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
            "max_tokens": 1024,
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
    log.info(f"  >> {decision.get('reasoning', 'N/A')[:150]}")

    for review in decision.get("position_review", []):
        log.info(f"  REVIEW {review['pair']}: {review['action']} — {review.get('reason', '')[:80]}")

    # Validate
    action = decision.get("action", "hold")
    if action == "hold":
        log.info("  => HOLD")
        return None

    confidence = decision.get("confidence", 0)
    if confidence < 0.65:
        log.info(f"  => Confidence {confidence:.2f} < 0.65 — forcing HOLD")
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
        "_position_reviews": decision.get("position_review", []),
    }

    return signal
