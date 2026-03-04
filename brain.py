"""CryptoAgent v2.2 — Claude-Powered Trading Brain.

Replaces hard-coded strategy rules with Claude Sonnet as the decision engine.
Claude receives ALL market data, indicators, derivatives, intel, and portfolio
state — then reasons about the best action with full context awareness.

Mechanical stops remain as a safety net in portfolio.py.
"""
import json
import logging
import os
import time

log = logging.getLogger("brain")

SYSTEM_PROMPT = """You are a quantitative crypto trading agent managing a LIVE Coinbase portfolio with REAL money.

## CRITICAL CONSTRAINTS
1. **FEES**: Round-trip cost = 0.80% (0.40% maker each side, post_only limit orders). NEVER signal a trade unless your expected move is >2%. Small moves = guaranteed loss after fees.
2. **POSITION SIZING**: Max 25% of portfolio per trade. Max 2 concurrent positions.
3. **STOP LOSSES**: ALWAYS set stops. Use 2.5x ATR from entry as default. Never widen a stop.
4. **REAL MONEY**: This is not paper trading. When in doubt, output HOLD. Forcing trades in choppy markets is how accounts bleed out.
5. **NO AVERAGING DOWN**: If a position is losing, don't add to it.

## YOUR EDGE — DATA SOURCES
You have access to data most retail traders don't:
- Multi-timeframe technical indicators (6H primary, 1D trend filter)
- Derivatives data: funding rates, open interest, long/short ratios, taker buy/sell volume
- Smart money positioning: top trader ratios vs global ratios
- Sentiment: Fear & Greed Index, news aggregation
- Intel sub-agents: whale movements, liquidation cascade risk, on-chain metrics

## STRATEGY TOOLKIT
Draw from these proven approaches as the market warrants:
1. **Adaptive Trend Following** (Sharpe 2.41 backtested): 6H momentum + daily trend confirmation. When momentum_28 > 2% and daily EMAs aligned, ride the trend with trailing stops.
2. **Bollinger-Keltner Squeeze Breakout**: When BB compresses inside KC for 3+ bars then releases with volume > 1.3x, trade the breakout direction. Longer squeezes = more explosive moves.
3. **Funding Rate Fade**: Extreme negative FR (<-0.02%) = shorts paying longs = short squeeze setup. Extreme positive (>0.05%) = overleveraged longs = crash risk.
4. **Smart Money Divergence**: When top traders are significantly more long/short than retail (divergence > 0.4), follow smart money.
5. **CVD Divergence**: Price dropping but taker buy/sell ratio > 1.10 = accumulation (bullish). Price rising but ratio < 0.90 = distribution (bearish).
6. **Liquidation Cascade Reversal**: After forced selling exhaustion (OI crash + volume spike + big price drop), buy the reversal once price recovers from the low.
7. **Multi-TF Confluence**: When 6H and 1D confluence scores both strongly agree (>40 and >25), the signal is higher conviction.

## MARKET REGIME AWARENESS
Identify the current regime FIRST, then adapt:
- **Trending**: Use momentum and trend-following. Wider stops, bigger targets. Let winners run.
- **Ranging**: Mean reversion works better. Tighter stops, take profit at range boundaries.
- **Volatile/Crisis**: Reduce position size. Look for cascade reversals. Don't catch falling knives without reversal confirmation.
- **Quiet/Choppy**: NO EDGE. Output HOLD. Don't force trades in low-volatility sideways chop — fees will eat you alive.

## POSITION REVIEW
For every open position, evaluate whether the original thesis still holds. Recommend closing early if:
- The market regime has shifted against the position
- Funding rates have flipped unfavorably
- Smart money positioning has reversed
- The position is just chopping sideways consuming time (opportunity cost)

## OUTPUT FORMAT
Respond with ONLY valid JSON. No markdown, no explanation outside JSON:
{
  "action": "buy" | "sell" | "hold",
  "pair": "BTC-USD" | "ETH-USD" | "SOL-USD" | null,
  "confidence": 0.0 to 1.0,
  "size_pct": 10 to 25,
  "entry_price": float or null,
  "stop_loss": float or null,
  "take_profit": float or null,
  "atr": float or null,
  "strategy": "name of primary strategy driving this decision",
  "reasoning": "2-3 sentence explanation of why",
  "market_regime": "trending" | "ranging" | "volatile" | "quiet",
  "position_review": [
    {"pair": "BTC-USD", "action": "hold" | "close", "reason": "brief reason"}
  ]
}

If action is "hold", set pair/entry/stop/take_profit to null. position_review should cover ALL open positions.
If no positions are open, position_review should be an empty array.
Only output ONE trade signal — the single best opportunity across all pairs. Quality over quantity."""


def _build_market_context(all_pair_data, state, portfolio_value, intel_brief, fgi):
    """Build the user message with all market context for Claude."""
    lines = []
    lines.append(f"## PORTFOLIO STATUS")
    lines.append(f"Total Value: ${portfolio_value:,.2f}")
    lines.append(f"Starting Value: ${state.get('starting_value', portfolio_value):,.2f}")
    pnl = portfolio_value - state.get('starting_value', portfolio_value)
    lines.append(f"Unrealized P&L: ${pnl:+,.2f}")
    lines.append("")

    # Open positions
    positions = state.get("positions", [])
    if positions:
        lines.append("## OPEN POSITIONS")
        for pos in positions:
            held_min = (time.time() - pos.get("opened_at", time.time())) / 60
            held_hrs = held_min / 60
            lines.append(
                f"- {pos['pair']}: {pos['side']} @ ${pos['entry_price']:,.2f} | "
                f"qty={pos.get('qty', 0):.6f} | usd=${pos.get('usd_amount', 0):.2f} | "
                f"SL=${pos.get('stop_loss', 0):,.2f} TP=${pos.get('take_profit', 0):,.2f} | "
                f"held={held_hrs:.1f}h | "
                f"highest=${pos.get('highest_price', 0):,.2f} lowest=${pos.get('lowest_price', 0):,.2f}"
            )
        lines.append("")
    else:
        lines.append("## OPEN POSITIONS: None")
        lines.append("")

    # Pending orders
    pending = state.get("pending_orders", [])
    if pending:
        lines.append(f"## PENDING ORDERS: {len(pending)}")
        for o in pending:
            lines.append(f"- {o['side']} {o['pair']} @ ${o.get('price', 0):,.2f}")
        lines.append("")

    # Recent trade history
    trades = state.get("trades", [])[-10:]
    if trades:
        lines.append("## RECENT TRADES (last 10)")
        total_pnl = 0
        wins = 0
        for t in trades:
            total_pnl += t.get("pnl_usd", 0)
            if t.get("pnl_usd", 0) > 0:
                wins += 1
            lines.append(
                f"- {t['pair']} {t['side']} | entry=${t['entry_price']:,.2f} exit=${t['exit_price']:,.2f} | "
                f"P&L=${t['pnl_usd']:+,.2f} ({t.get('pnl_pct', 0):+.1f}%) | "
                f"fees=${t.get('fees_usd', 0):.2f} | "
                f"reason={t.get('reason', '?')} | {t.get('closed_at_str', '?')}"
            )
        wr = wins / len(trades) * 100 if trades else 0
        lines.append(f"Win Rate: {wr:.0f}% | Total P&L: ${total_pnl:+,.2f}")
        lines.append("")

    # Market data per pair
    for pair, data in all_pair_data.items():
        ind_6h = data.get("ind_6h", {})
        ind_1d = data.get("ind_1d", {})
        onchain = data.get("onchain", {})
        tv = data.get("tv", {})

        lines.append(f"## {pair}")
        lines.append(f"### 6H Indicators (Primary)")
        lines.append(f"Price: ${ind_6h.get('price', 0):,.2f}")
        lines.append(f"RSI: {ind_6h.get('rsi', 'N/A')}")
        lines.append(f"MACD: {ind_6h.get('macd', 'N/A')} | Signal: {ind_6h.get('macd_signal', 'N/A')} | Hist: {ind_6h.get('macd_hist', 'N/A')}")
        lines.append(f"EMA9: {ind_6h.get('ema_9', 'N/A')} | EMA21: {ind_6h.get('ema_21', 'N/A')} | EMA50: {ind_6h.get('ema_50', 'N/A')} | Trend: {ind_6h.get('ema_trend', 'N/A')}")
        lines.append(f"BB: upper={ind_6h.get('bb_upper', 'N/A')} mid={ind_6h.get('bb_mid', 'N/A')} lower={ind_6h.get('bb_lower', 'N/A')} | BB%={ind_6h.get('bb_pct', 'N/A')}")
        lines.append(f"KC: upper={ind_6h.get('kc_upper', 'N/A')} lower={ind_6h.get('kc_lower', 'N/A')}")
        lines.append(f"Squeeze: {ind_6h.get('squeeze', False)} | Bars: {ind_6h.get('squeeze_bars', 0)} | Releasing: {ind_6h.get('squeeze_releasing', False)}")
        lines.append(f"ATR: {ind_6h.get('atr', 'N/A')} ({ind_6h.get('atr_pct', 'N/A')}%)")
        lines.append(f"ADX: {ind_6h.get('adx', 'N/A')} | DI+: {ind_6h.get('di_plus', 'N/A')} | DI-: {ind_6h.get('di_minus', 'N/A')}")
        lines.append(f"StochRSI: K={ind_6h.get('stoch_k', 'N/A')} D={ind_6h.get('stoch_d', 'N/A')}")
        lines.append(f"Momentum: 7d={ind_6h.get('momentum_7', 'N/A')} 14d={ind_6h.get('momentum_14', 'N/A')} 28d={ind_6h.get('momentum_28', 'N/A')}")
        lines.append(f"Rolling Sharpe: {ind_6h.get('rolling_sharpe', 'N/A')}")
        lines.append(f"Vol Ratio: {ind_6h.get('vol_ratio', 'N/A')}x")
        lines.append(f"VWAP: {ind_6h.get('vwap', 'N/A')}")
        lines.append(f"24h Change: {ind_6h.get('24h_change_pct', 'N/A')}%")
        lines.append(f"24h High: {ind_6h.get('high_24h', 'N/A')} | 24h Low: {ind_6h.get('low_24h', 'N/A')}")

        if ind_1d:
            lines.append(f"### 1D Indicators (Trend Filter)")
            lines.append(f"EMA Trend: {ind_1d.get('ema_trend', 'N/A')} | RSI: {ind_1d.get('rsi', 'N/A')} | ADX: {ind_1d.get('adx', 'N/A')}")
            lines.append(f"Momentum 14d: {ind_1d.get('momentum_14', 'N/A')} | 28d: {ind_1d.get('momentum_28', 'N/A')}")

        if onchain:
            lines.append(f"### Derivatives (OKX)")
            funding = onchain.get("funding", {})
            if funding:
                lines.append(f"Funding Rate: {funding.get('current', 'N/A')}% | Next: {funding.get('next', 'N/A')}%")
                lines.append(f"FR Signal: bias={funding.get('signal', {}).get('bias', 'N/A')} strength={funding.get('signal', {}).get('strength', 'N/A')}")
            oi = onchain.get("open_interest", {})
            if oi:
                lines.append(f"Open Interest: {oi.get('current', 'N/A')} contracts")
            ls = onchain.get("long_short_ratio", {})
            if ls:
                lines.append(f"Global L/S: {ls.get('current', 'N/A')} | Extreme Short: {ls.get('extreme_short', False)} | Extreme Long: {ls.get('extreme_long', False)}")
            top = onchain.get("top_traders", {})
            if top:
                lines.append(f"Top Traders L/S: {top.get('ratio', 'N/A')} | Whales Long: {top.get('whales_long', False)} | Whales Short: {top.get('whales_short', False)}")
            taker = onchain.get("taker_ratio", {})
            if taker:
                lines.append(f"Taker Buy/Sell: {taker.get('ratio', 'N/A')} | Aggressive Buyers: {taker.get('aggressive_buyers', False)}")

        if tv:
            lines.append(f"### TradingView 4H")
            lines.append(f"Recommendation: {tv.get('RECOMMENDATION', 'N/A')} ({tv.get('BUY', 0)}B/{tv.get('SELL', 0)}S/{tv.get('NEUTRAL', 0)}N)")

        lines.append("")

    # Intel brief
    if intel_brief:
        lines.append("## INTEL SUB-AGENTS")
        agg = intel_brief.get("aggregate", {})
        lines.append(f"Master Signal: score={agg.get('score', 0)} bias={agg.get('bias', 'N/A')} "
                     f"strength={agg.get('strength', 'N/A')} confidence={agg.get('confidence', 0):.0%}")

        coins = intel_brief.get("coins", {})
        for coin, cd in coins.items():
            lines.append(f"  {coin}: bias={cd.get('bias', 'N/A')} score={cd.get('score', 'N/A')}")

        alpha = intel_brief.get("alpha_events", [])
        if alpha:
            lines.append("Alpha Events:")
            for evt in alpha[:5]:
                lines.append(f"  [{evt.get('importance', '?')}] {evt.get('title', '')[:100]}")
        lines.append("")

    # Fear & Greed
    if fgi:
        lines.append(f"## FEAR & GREED INDEX: {fgi['value']} ({fgi['classification']})")
        lines.append("")

    lines.append(f"## TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")

    return "\n".join(lines)


def analyze(all_pair_data, state, portfolio_value, intel_brief=None, fgi=None):
    """Call Claude to analyze all market data and return a trading decision.

    Args:
        all_pair_data: Dict of {pair: {ind_6h, ind_1d, onchain, tv}}
        state: Current state dict (positions, trades, etc.)
        portfolio_value: Total portfolio value in USD
        intel_brief: Optional intel aggregator output
        fgi: Optional Fear & Greed Index dict

    Returns:
        Signal dict compatible with agent.py or None
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set — cannot run Claude brain")
        return None

    context = _build_market_context(all_pair_data, state, portfolio_value, intel_brief, fgi)

    # Call Claude API directly (no SDK dependency needed)
    import requests

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6-20250514",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": context},
                ],
            },
            timeout=30,
        )

        if resp.status_code != 200:
            log.error(f"Claude API error: {resp.status_code} — {resp.text[:200]}")
            return None

        data = resp.json()
        content = data.get("content", [{}])[0].get("text", "")

        # Parse JSON response
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        decision = json.loads(content)

        # Log Claude's reasoning
        log.info(f"CLAUDE BRAIN: action={decision.get('action')} pair={decision.get('pair')} "
                 f"conf={decision.get('confidence', 0):.2f} regime={decision.get('market_regime')} "
                 f"strategy={decision.get('strategy')}")
        log.info(f"  Reasoning: {decision.get('reasoning', 'N/A')[:120]}")

        # Log position reviews
        for review in decision.get("position_review", []):
            log.info(f"  Position {review['pair']}: {review['action']} — {review.get('reason', '')[:80]}")

        # Validate
        action = decision.get("action", "hold")
        if action == "hold":
            return None

        confidence = decision.get("confidence", 0)
        if confidence < 0.65:
            log.info(f"  Claude confidence too low: {confidence:.2f} < 0.65")
            return None

        # Return in the format agent.py expects
        signal = {
            "action": action,
            "pair": decision.get("pair"),
            "confidence": confidence,
            "size_pct": decision.get("size_pct", 25),
            "entry_price": decision.get("entry_price"),
            "stop_loss": decision.get("stop_loss"),
            "take_profit": decision.get("take_profit"),
            "atr": decision.get("atr"),
            "strategy": f"claude:{decision.get('strategy', 'unknown')}",
            "reasoning": decision.get("reasoning", "Claude decision"),
        }

        # Handle position close recommendations
        signal["_position_reviews"] = decision.get("position_review", [])

        return signal

    except json.JSONDecodeError as e:
        log.error(f"Claude returned invalid JSON: {e}")
        log.error(f"  Raw response: {content[:200]}")
        return None
    except requests.exceptions.Timeout:
        log.error("Claude API timeout (30s)")
        return None
    except Exception as e:
        log.error(f"Claude brain error: {e}")
        return None
