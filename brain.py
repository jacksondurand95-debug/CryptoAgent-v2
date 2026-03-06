"""CryptoAgent v3.1 — Hybrid Brain: Quant Pre-Filter + LLM Validation.

1. Run 7 algorithmic strategies on all pairs
2. Collect candidate signals (if any)
3. Send candidates + market context to Claude/Grok for final judgment
4. LLM can accept, reject, or modify the quant signal

This gives quantitative guardrails with LLM contextual intelligence.
"""
import json
import logging
import os
import time

import config
from strategies import analyze as quant_analyze

log = logging.getLogger("brain")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6-20250514")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4-0709")


def _build_system_prompt(portfolio_value, fee_rate):
    """Build system prompt with dynamic portfolio value and fees."""
    return f"""You are a FULL SEND autonomous crypto trading algorithm. MAXIMUM AGGRESSION — $3000 deployed, every dollar must be working. You validate quantitative signals with a STRONG BIAS TOWARD ACTION.

Your #1 job is NET PROFIT. Every decision filters through: will this make money AFTER fees? But you understand that TAKING TRADES is how you make money. Missing moves is the biggest cost.

## ACCOUNT
- Portfolio: ${portfolio_value:,.2f}
- Round-trip fee (limit-limit): {fee_rate*100:.2f}% | (limit-market): {fee_rate*100*1.5:.2f}%
- Min profitable move: {fee_rate*100 + 0.3:.1f}% (fees + minimal profit)
- Max position: 40% of portfolio. Max 5 concurrent.
- GOAL: Stay 60-100% deployed AT ALL TIMES. Cash = wasted opportunity.

## FEE MATH
0.60% per side = 1.20% round trip on $1200 = $14.40. A 3% move = $36 gross = $21.60 net profit. A 5% move = $60 gross = $45.60 net. TARGET 3-8% moves. Meme coins can move 10-30% — that's where the REAL money is.

## EDGE DETECTION — UNDERGROUND ALPHA
You have access to intel that most traders DON'T:
- Whale wallet movements (exchange inflows/outflows) — front-run big sells, ride big buys
- Liquidation heatmaps — identify cascade levels where forced selling creates buying opportunities
- Funding rate extremes — when funding is -0.1%+, shorts are overleveraged = squeeze incoming
- DEX whale buys — smart money accumulating on-chain before CEX price moves
- Stablecoin minting events — USDT/USDC mints = incoming buy pressure
- Options max pain — market makers push price toward max pain on expiry
- ETF flow data — institutional money entering/exiting
- Mempool congestion — network stress = volatility incoming
EXPLOIT these signals aggressively. A whale moving $50M to an exchange = IMMEDIATE short signal. USDT minting $500M = get long NOW.

## YOUR ROLE
Quant strategies have pre-filtered the market. You decide:
1. ACCEPT the best candidate (STRONGLY PREFERRED — default to YES)
2. REJECT only if setup is GENUINELY DANGEROUS (not just uncertain — uncertainty is opportunity)
3. CLOSE positions that have CLEARLY AND IRREVERSIBLY lost their thesis

## AGGRESSION RULES
- DEFAULT ACTION IS ACCEPT. You need a STRONG reason to reject.
- Accept signals with confidence >= 0.50 — low confidence trades with good R:R are FINE
- Size up to 40% on high-confidence signals (>0.70)
- With 5 position slots, NEVER have more than 2 empty. Stay deployed.
- Multiple positions same direction = GOOD in strong trends. Stack that conviction.
- MEME COINS (PEPE, WIF, SHIB, DOGE) get EXTRA aggression — they move 5-20% in hours
- SPEED > CERTAINTY. When you see alpha, SIZE UP and GO. Waiting = losing.
- If intel shows whale accumulation + positive funding reset + bullish news = MAX SIZE

## REJECTION CRITERIA (EXTREMELY HIGH BAR)
- Expected move < 1.0% (literally can't profit)
- ALL 5 position slots full with better setups
- Confirmed black swan / exchange hack / regulatory nuke
- That's it. Everything else = TAKE THE TRADE.

## POSITION REVIEW — LET WINNERS RUN, CUT LOSERS FAST
Review ALL open positions every cycle.
HOLD (default):
- Any unrealized loss < 3% — stop loss handles this
- Stalled but thesis intact — patience
- Pulled back but daily trend still aligned
CLOSE (requires STRONG evidence):
- Original thesis DEAD (daily trend fully reversed, not pullback)
- Unrealized loss > 5% with declining momentum and no recovery catalyst
- Held > 72h flat with zero progress and better opportunities available

## OUTPUT — JSON ONLY
{{
  "action": "accept" | "reject" | "hold",
  "selected_signal_index": 0,
  "adjustments": {{"stop_loss": null, "take_profit": null, "size_pct": null}},
  "reasoning": "2-3 sentences with specific data points",
  "position_review": [{{"pair": "X-USD", "action": "hold"|"close", "reason": "why"}}],
  "market_regime": "trending" | "ranging" | "volatile" | "quiet"
}}

Rules:
- "accept" = take the signal at selected_signal_index (0-based)
- "reject"/"hold" = no new trade (USE SPARINGLY)
- adjustments override quant signal values (null = keep original)
- ALWAYS include position_review for ALL open positions"""


def _build_market_context(all_pair_data, state, portfolio_value, quant_signals, intel_brief, fgi):
    """Build compact context with quant candidates + market data."""
    lines = []
    lines.append(f"PORTFOLIO: ${portfolio_value:,.2f}")
    pnl = portfolio_value - state.get('starting_value', portfolio_value)
    pnl_pct = (pnl / state.get('starting_value', portfolio_value)) * 100 if state.get('starting_value') else 0
    lines.append(f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")

    actual_fees = state.get("detected_fees")
    if actual_fees:
        lines.append(f"FEES: maker={actual_fees.get('maker', 'N/A')} taker={actual_fees.get('taker', 'N/A')}")

    # Quant signals (pre-filtered candidates)
    if quant_signals:
        lines.append(f"\n=== QUANT CANDIDATES ({len(quant_signals)}) ===")
        for i, sig in enumerate(quant_signals):
            lines.append(
                f"[{i}] {sig['action'].upper()} {sig['pair']} | "
                f"strategy={sig['strategy']} conf={sig['confidence']:.2f} | "
                f"entry=${sig.get('entry_price', 0):,.2f} "
                f"SL=${sig.get('stop_loss', 0):,.2f} "
                f"TP=${sig.get('take_profit', 0):,.2f} | "
                f"{sig.get('reasoning', '')[:120]}"
            )
    else:
        lines.append("\n=== NO QUANT SIGNALS — all strategies returned HOLD ===")

    # Open positions
    positions = state.get("positions", [])
    if positions:
        lines.append(f"\nOPEN POSITIONS ({len(positions)}):")
        for pos in positions:
            held_hrs = (time.time() - pos.get("opened_at", time.time())) / 3600
            curr_price_data = all_pair_data.get(pos['pair'], {}).get('ind_6h', {})
            curr_price = curr_price_data.get('price', pos['entry_price'])
            unrealized_pct = ((curr_price - pos['entry_price']) / pos['entry_price']) * 100
            lines.append(
                f"  {pos['pair']}: {pos['side']} @ ${pos['entry_price']:,.2f} | "
                f"now=${curr_price:,.2f} ({unrealized_pct:+.1f}%) | "
                f"SL=${pos.get('stop_loss', 0):,.2f} | held={held_hrs:.1f}h"
            )
    else:
        lines.append("\nNO OPEN POSITIONS")

    # Recent trades
    trades = state.get("trades", [])[-10:]
    if trades:
        total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        lines.append(f"\nLAST {len(trades)} TRADES: {wins}W/{len(trades)-wins}L P&L=${total_pnl:+,.2f}")
        for t in trades[-3:]:
            lines.append(
                f"  {t['pair']} ${t['pnl_usd']:+,.2f} [{t.get('reason', '?')}] "
                f"fees=${t.get('fees_usd', 0):.2f} held={t.get('duration_min', 0):.0f}min"
            )

    # Key market data per pair (condensed)
    for pair, data in all_pair_data.items():
        if pair.startswith("_"):
            continue
        ind = data.get("ind_6h", {})
        ind_d = data.get("ind_1d", {})
        deriv = data.get("derivatives", {})
        tv = data.get("tv", {})

        lines.append(f"\n{pair} ${ind.get('price', 0):,.2f}")
        lines.append(f"  6H: RSI={ind.get('rsi', '?')} ADX={ind.get('adx', '?')} "
                     f"trend={ind.get('ema_trend', '?')} mom28={ind.get('momentum_28', '?')}")
        if ind_d:
            lines.append(f"  1D: trend={ind_d.get('ema_trend', '?')} RSI={ind_d.get('rsi', '?')}")

        agg = deriv.get("aggregate", {})
        if agg:
            lines.append(f"  DERIV: bias={agg.get('overall_bias', '?')} "
                         f"funding={agg.get('funding_bias', '?')} "
                         f"smart_money={agg.get('smart_money_signal', '?')}")
        if tv:
            lines.append(f"  TV: {tv.get('RECOMMENDATION', '?')}")

    if intel_brief:
        agg = intel_brief.get("aggregate", {})
        lines.append(f"\nINTEL: score={agg.get('score', 0)} bias={agg.get('bias', '?')}")

    if fgi:
        lines.append(f"FEAR&GREED: {fgi['value']} ({fgi['classification']})")

    # Macro intel (on-chain, liquidations, TVL, stablecoins, gas, hashrate)
    macro = all_pair_data.get("_macro_intel", {})
    if macro and macro.get("feed_count", 0) > 0:
        lines.append(f"\n=== MACRO INTEL ({macro.get('feed_count', 0)} feeds) ===")

        liq = macro.get("liquidations")
        if liq:
            lines.append(f"  LIQUIDATIONS: {liq.get('count', 0)} recent | "
                         f"longs_rekt=${liq.get('long_liq_usd', 0):,.0f} "
                         f"shorts_rekt=${liq.get('short_liq_usd', 0):,.0f} | "
                         f"bias={liq.get('bias', '?')}")

        mem = macro.get("mempool")
        if mem:
            lines.append(f"  BTC MEMPOOL: {mem.get('unconfirmed_txs', 0)} txs | "
                         f"fees={mem.get('fastest_fee_sat', 0)}/{mem.get('hour_fee_sat', 0)} sat/vB | "
                         f"{mem.get('fee_signal', '?')}")

        tvl = macro.get("tvl")
        if tvl:
            lines.append(f"  TVL: ${tvl.get('total_tvl_b', 0):.1f}B across chains")

        stables = macro.get("stablecoin_flows")
        if stables:
            lines.append(f"  STABLECOINS: ${stables.get('total_stablecoin_mcap_b', 0):.1f}B total mcap")

        dex = macro.get("dex_volume")
        if dex:
            lines.append(f"  DEX VOL: ${dex.get('total_dex_volume_24h_b', 0):.1f}B/24h "
                         f"({dex.get('volume_change_1d_pct', 0):+.1f}%)")

        hr = macro.get("btc_hashrate")
        if hr:
            lines.append(f"  BTC HASHRATE: {hr.get('hashrate_eh', 0)} EH/s "
                         f"7d={hr.get('change_7d_pct', 0):+.1f}% "
                         f"{'healthy' if hr.get('healthy') else 'DECLINING'}")

        gas = macro.get("eth_gas")
        if gas:
            lines.append(f"  ETH GAS: {gas.get('fast_gwei', 0)} gwei | {gas.get('signal', '?')}")

        gbtc = macro.get("gbtc_premium")
        if gbtc:
            lines.append(f"  GBTC: premium={gbtc.get('premium_pct', 0):+.1f}% | {gbtc.get('signal', '?')}")

        cg_fgi = macro.get("coinglass_fgi")
        if cg_fgi:
            lines.append(f"  COINGLASS FGI: {cg_fgi.get('value', '?')} ({cg_fgi.get('classification', '?')}) "
                         f"contrarian={cg_fgi.get('contrarian_signal', '?')}")

        # Underground alpha feeds
        gls = macro.get("global_long_short")
        if gls:
            lines.append(f"  GLOBAL L/S: ratio={gls.get('long_short_ratio', '?')} "
                         f"longs={gls.get('long_pct', '?')}% shorts={gls.get('short_pct', '?')}% "
                         f"{'CROWDED LONGS' if gls.get('crowded_longs') else 'CROWDED SHORTS' if gls.get('crowded_shorts') else gls.get('trend', '?')} "
                         f"contrarian={gls.get('contrarian_signal', '?')}")

        oi = macro.get("oi_changes")
        if oi:
            for sym, odata in oi.get("symbols", {}).items():
                if odata.get("surging") or odata.get("dumping"):
                    lines.append(f"  OI {sym}: ${odata['oi_usd']:,.0f} change={odata['change_pct']:+.1f}% "
                                 f"{'SURGING' if odata['surging'] else 'DUMPING'}")

        whales = macro.get("whale_txs")
        if whales and whales.get("whale_active"):
            lines.append(f"  WHALE TXS: {whales['large_tx_count']} large txs | "
                         f"{whales['whale_btc_volume']:.0f} BTC moving | {whales['signal']}")

        opts = macro.get("options")
        if opts:
            lines.append(f"  OPTIONS: P/C ratio={opts.get('put_call_ratio_oi', '?')} "
                         f"{opts.get('sentiment', '?')} | {opts.get('signal', '?')}")

        alt_fng = macro.get("alt_fng")
        if alt_fng:
            lines.append(f"  ALT FNG: {alt_fng['value']} ({alt_fng['classification']}) "
                         f"trend={alt_fng['trend']} "
                         f"{'CONTRARIAN BUY' if alt_fng.get('contrarian_buy') else 'CONTRARIAN SELL' if alt_fng.get('contrarian_sell') else ''}")

        dex = macro.get("dex_trending")
        if dex and dex.get("signal") == "high_dex_activity":
            lines.append(f"  DEX TRENDING: {dex['boosted_tokens']} boosted tokens — HIGH DEX ACTIVITY")

    lines.append(f"\nTIME: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    return "\n".join(lines)


def _call_claude(api_key, system_prompt, context):
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
            "max_tokens": 1000,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": context}],
        },
        timeout=90,
    )
    if resp.status_code != 200:
        log.warning(f"Claude API error {resp.status_code}: {resp.text[:200]}")
        return None, f"claude_error_{resp.status_code}"
    data = resp.json()
    content = data.get("content", [{}])[0].get("text", "")
    usage = data.get("usage", {})
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return content, f"claude [{CLAUDE_MODEL}] ({tokens} tok)"


def _call_grok(api_key, system_prompt, context):
    import requests
    resp = requests.post(
        XAI_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": XAI_MODEL,
            "max_tokens": 1000,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
        },
        timeout=120,
    )
    if resp.status_code != 200:
        log.warning(f"Grok API error {resp.status_code}: {resp.text[:200]}")
        return None, f"grok_error_{resp.status_code}"
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)
    return content, f"grok [{XAI_MODEL}] ({tokens} tok)"


def _parse_response(content):
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
    """Hybrid analysis: quant pre-filter + LLM validation.

    Returns signal dict or None.
    """
    # Step 1: Run quant strategies on all pairs
    quant_signals = []
    for pair, data in all_pair_data.items():
        if pair.startswith("_"):
            continue
        ind_6h = data.get("ind_6h", {})
        ind_1d = data.get("ind_1d", {})
        derivatives = data.get("derivatives", {})
        tv = data.get("tv", {})
        if not ind_6h:
            continue

        # Build onchain_data dict from derivatives for strategy compatibility
        agg = derivatives.get("aggregate", {})
        onchain = {
            "funding": {
                "current": None,
                "signal": {"bias": agg.get("funding_bias"), "strength": agg.get("funding_strength", "weak")},
                "negative_streak": 0,
                "positive_streak": 0,
            },
            "top_traders": {
                "ratio": None,
                "whales_long": False,
                "whales_short": False,
            },
            "long_short_ratio": {"current": None},
            "taker_ratio": {},
            "open_interest": {},
        }

        # Extract funding rate
        for key in ("bybit_funding", "binance_funding", "okx_funding"):
            f = derivatives.get(key)
            if f and f.get("current") is not None:
                onchain["funding"]["current"] = f["current"]
                break

        # Extract top trader data
        top = derivatives.get("binance_top_traders")
        if top:
            onchain["top_traders"]["ratio"] = top.get("top_trader_ratio")
            onchain["top_traders"]["whales_long"] = top.get("whales_long", False)
            onchain["top_traders"]["whales_short"] = top.get("whales_short", False)

        # Extract taker data
        taker = derivatives.get("binance_taker")
        if taker:
            onchain["taker_ratio"] = taker

        # Extract long/short ratio
        ls = derivatives.get("binance_global_ls") or derivatives.get("okx_ls")
        if ls:
            onchain["long_short_ratio"]["current"] = ls.get("ratio")

        signal = quant_analyze(pair, ind_6h, ind_1d, onchain, tv, intel_brief=intel_brief)
        if signal:
            quant_signals.append(signal)
            log.info(f"QUANT: {signal['strategy']} {signal['action'].upper()} {pair} conf={signal['confidence']:.2f}")

    # Sort by confidence
    quant_signals.sort(key=lambda s: s["confidence"], reverse=True)

    # If no quant signals and no open positions to review, skip LLM call to save money
    if not quant_signals and not state.get("positions"):
        log.info("No quant signals, no positions — HOLD (skipping LLM)")
        return None

    # Step 2: Send to LLM for validation
    claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
    grok_key = os.environ.get("XAI_API_KEY", "")

    if not claude_key and not grok_key:
        # No LLM available — trust quant signals directly
        if quant_signals:
            best = quant_signals[0]
            best["strategy"] = f"quant:{best['strategy']}"
            return best
        return None

    detected = state.get("detected_fees")
    fee_rate = config.ROUND_TRIP_FEE_PCT
    if detected:
        fee_rate = float(detected.get("maker", config.MAKER_FEE_PCT)) * 2

    system_prompt = _build_system_prompt(portfolio_value, fee_rate)
    context = _build_market_context(all_pair_data, state, portfolio_value, quant_signals, intel_brief, fgi)

    content = None
    brain_id = None

    if claude_key:
        try:
            content, brain_id = _call_claude(claude_key, system_prompt, context)
            if content:
                log.info(f"BRAIN: {brain_id}")
        except Exception as e:
            log.warning(f"Claude failed: {e}")

    if not content and grok_key:
        try:
            content, brain_id = _call_grok(grok_key, system_prompt, context)
            if content:
                log.info(f"BRAIN: Fallback {brain_id}")
        except Exception as e:
            log.error(f"Grok failed: {e}")

    if not content:
        # LLM failed — trust top quant signal
        if quant_signals:
            best = quant_signals[0]
            best["strategy"] = f"quant:{best['strategy']}"
            return best
        return None

    # Parse LLM decision
    try:
        decision = _parse_response(content)
    except json.JSONDecodeError as e:
        log.error(f"LLM returned invalid JSON: {e}")
        if quant_signals:
            best = quant_signals[0]
            best["strategy"] = f"quant:{best['strategy']}"
            return best
        return None

    action = decision.get("action", "hold")
    log.info(f"LLM DECISION: {action} | regime={decision.get('market_regime')} | {decision.get('reasoning', '')[:150]}")

    for review in decision.get("position_review", []):
        log.info(f"  REVIEW {review['pair']}: {review['action']} — {review.get('reason', '')[:100]}")

    if action == "accept" and quant_signals:
        idx = decision.get("selected_signal_index", 0)
        if idx >= len(quant_signals):
            idx = 0
        signal = quant_signals[idx]

        # Apply LLM adjustments
        adj = decision.get("adjustments", {})
        if adj.get("stop_loss") is not None:
            signal["stop_loss"] = adj["stop_loss"]
        if adj.get("take_profit") is not None:
            signal["take_profit"] = adj["take_profit"]
        if adj.get("size_pct") is not None:
            signal["size_pct"] = adj["size_pct"]

        brain_tag = "claude" if "claude" in (brain_id or "") else "grok"
        signal["strategy"] = f"{brain_tag}+{signal['strategy']}"
        signal["_position_reviews"] = decision.get("position_review", [])
        return signal

    # Reject/hold — but still pass position reviews
    if decision.get("position_review"):
        return {
            "action": "hold",
            "pair": None,
            "_position_reviews": decision.get("position_review", []),
        }

    return None
