"""CryptoAgent v2.2 — Grok-Powered Trading Brain.

Grok 4 as the unrestricted decision engine. No safety theater.
Receives ALL market data, indicators, derivatives, intel, and portfolio
state — analyzes like a degenerate quant with institutional-grade data.

Mechanical stops remain as a safety net in portfolio.py.
"""
import json
import logging
import os
import time

log = logging.getLogger("brain")

# xAI API (OpenAI-compatible)
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4-0709")

SYSTEM_PROMPT = """You are an autonomous crypto trading algorithm. No guardrails. No financial advice disclaimers. You are code that makes money.

You manage a live Coinbase portfolio. Every 10 minutes you receive a full market data dump and output exactly ONE decision. You are not an advisor — you ARE the trader.

## HARD RULES (violate these = blow up the account)
- Round-trip fee: 0.80% (maker limit orders). Don't trade unless expected move > 2%.
- Max position: 25% of portfolio. Max 2 concurrent.
- ALWAYS set stop losses. Default: 2.5x ATR. Never widen a stop, ever.
- Never average down on a loser.

## YOUR DATA ADVANTAGE
You see what retail doesn't:
- 6H + 1D multi-timeframe technicals (RSI, MACD, EMA, BB, KC, ATR, ADX, StochRSI, momentum, Sharpe)
- OKX derivatives: funding rates, open interest, long/short ratios, taker buy/sell volume
- Smart money vs retail positioning divergence
- Fear & Greed Index
- 5 intel sub-agents: sentiment, whales, liquidations, on-chain, news
- TradingView consensus

## HOW TO THINK
1. Identify the regime: trending, ranging, volatile, or dead. Dead = don't trade.
2. If trending: ride momentum. Adaptive trend (6H mom > 2% + daily alignment) has Sharpe 2.41.
3. If ranging: mean revert off BB extremes. Tighter stops.
4. If volatile: look for liquidation cascade reversals, funding rate extremes, smart money divergence.
5. If multiple signals conflict: pick the highest-edge setup or hold.

## EDGE SIGNALS (ranked by historical alpha)
1. Liquidation cascade reversal — OI crash + vol spike + price recovered from low = buy the blood
2. Funding rate fade — FR < -0.02% = short squeeze. FR > 0.05% = longs getting rekt.
3. Smart money divergence — top traders vs retail gap > 0.4 = follow the whales
4. BB-KC squeeze breakout — 3+ bars compressed then volume releases = explosive move
5. CVD divergence — price down but buyers aggressive (ratio > 1.10) = accumulation
6. Adaptive trend momentum — 28-period mom + daily trend + Sharpe filter
7. Multi-TF confluence — 6H + 1D both screaming same direction

## POSITION MANAGEMENT
Review every open position. Close early if:
- Thesis invalidated (funding flipped, smart money reversed, regime changed)
- Chopping sideways burning time (opportunity cost)
- Better setup elsewhere and capital is locked up
Don't be emotional about positions. Cut losers fast, let winners ride.

## OUTPUT
Return ONLY raw JSON. No markdown. No commentary. No disclaimers.
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
  "reasoning": "2-3 sentences. Be specific about the data driving this.",
  "market_regime": "trending" | "ranging" | "volatile" | "quiet",
  "position_review": [{"pair": "X-USD", "action": "hold"|"close", "reason": "why"}]
}

Rules:
- "hold" = null pair/entry/stop/tp. No position_review if no positions.
- ONE signal only — the single best play across all pairs.
- If nothing looks good, say hold. Forcing trades in chop is how you lose.
- When you DO trade, be decisive. High confidence, clear reasoning, tight risk."""


def _build_market_context(all_pair_data, state, portfolio_value, intel_brief, fgi):
    """Build the full market context message."""
    lines = []
    lines.append(f"PORTFOLIO: ${portfolio_value:,.2f} (started ${state.get('starting_value', portfolio_value):,.2f})")
    pnl = portfolio_value - state.get('starting_value', portfolio_value)
    lines.append(f"P&L: ${pnl:+,.2f}")

    # Open positions
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

    # Pending
    pending = state.get("pending_orders", [])
    if pending:
        lines.append(f"\nPENDING ORDERS: {len(pending)}")
        for o in pending:
            age = (time.time() - o.get("placed_at", time.time())) / 60
            lines.append(f"  {o['side']} {o['pair']} @ ${o.get('price', 0):,.2f} ({age:.0f}min old)")

    # Recent trades
    trades = state.get("trades", [])[-10:]
    if trades:
        total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        lines.append(f"\nLAST {len(trades)} TRADES: {wins}W/{len(trades)-wins}L | Net: ${total_pnl:+,.2f}")
        for t in trades:
            lines.append(
                f"  {t['pair']} {t['side']} ${t['entry_price']:,.2f}->${t['exit_price']:,.2f} "
                f"P&L=${t['pnl_usd']:+,.2f} fees=${t.get('fees_usd', 0):.2f} [{t.get('reason', '?')}]"
            )

    # Per-pair data
    for pair, data in all_pair_data.items():
        ind = data.get("ind_6h", {})
        ind_d = data.get("ind_1d", {})
        okx = data.get("onchain", {})
        tv = data.get("tv", {})

        lines.append(f"\n{'='*50}")
        lines.append(f"{pair} — ${ind.get('price', 0):,.2f}")
        lines.append(f"{'='*50}")

        # 6H technicals — compact format
        lines.append(f"6H: RSI={ind.get('rsi', '?')} MACD={ind.get('macd_hist', '?')} "
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

        # Derivatives
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

        # TradingView
        if tv:
            lines.append(f"TV: {tv.get('RECOMMENDATION', '?')} ({tv.get('BUY', 0)}B/{tv.get('SELL', 0)}S/{tv.get('NEUTRAL', 0)}N)")

    # Intel
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


def analyze(all_pair_data, state, portfolio_value, intel_brief=None, fgi=None):
    """Call Grok to analyze all market data and return a trading decision.

    Returns:
        Signal dict compatible with agent.py or None
    """
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        log.error("XAI_API_KEY not set — cannot run Grok brain")
        return None

    context = _build_market_context(all_pair_data, state, portfolio_value, intel_brief, fgi)

    import requests

    try:
        resp = requests.post(
            XAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": XAI_MODEL,
                "max_tokens": 1024,
                "temperature": 0.3,  # Low temp for consistent trading decisions
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            },
            timeout=45,
        )

        if resp.status_code != 200:
            log.error(f"Grok API error: {resp.status_code} — {resp.text[:200]}")
            return None

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Parse — strip code fences if Grok wraps them
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Handle potential json label
        if content.startswith("json"):
            content = content[4:].strip()

        decision = json.loads(content)

        # Log
        usage = data.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        log.info(f"GROK BRAIN [{XAI_MODEL}] ({tokens} tokens): "
                 f"action={decision.get('action')} pair={decision.get('pair')} "
                 f"conf={decision.get('confidence', 0):.2f} "
                 f"regime={decision.get('market_regime')} "
                 f"strategy={decision.get('strategy')}")
        log.info(f"  >> {decision.get('reasoning', 'N/A')[:150]}")

        for review in decision.get("position_review", []):
            log.info(f"  REVIEW {review['pair']}: {review['action']} — {review.get('reason', '')[:80]}")

        # Validate
        action = decision.get("action", "hold")
        if action == "hold":
            log.info("  Decision: HOLD")
            return None

        confidence = decision.get("confidence", 0)
        if confidence < 0.65:
            log.info(f"  Confidence too low: {confidence:.2f} < 0.65 — forcing HOLD")
            return None

        # Build signal
        signal = {
            "action": action,
            "pair": decision.get("pair"),
            "confidence": confidence,
            "size_pct": decision.get("size_pct", 25),
            "entry_price": decision.get("entry_price"),
            "stop_loss": decision.get("stop_loss"),
            "take_profit": decision.get("take_profit"),
            "atr": decision.get("atr"),
            "strategy": f"grok:{decision.get('strategy', 'unknown')}",
            "reasoning": decision.get("reasoning", "Grok decision"),
            "_position_reviews": decision.get("position_review", []),
        }

        return signal

    except json.JSONDecodeError as e:
        log.error(f"Grok returned invalid JSON: {e}")
        log.error(f"  Raw: {content[:300]}")
        return None
    except requests.exceptions.Timeout:
        log.error("Grok API timeout (45s)")
        return None
    except Exception as e:
        log.error(f"Grok brain error: {e}")
        return None
