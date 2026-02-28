"""CryptoAgent v2 — 7 Ultra Aggressive Trading Strategies.

Each strategy takes (pair, indicators_1h, onchain_data, tv_analysis)
and returns a signal dict or None.

Signal dict format:
{
    "action": "buy" or "sell",
    "pair": "BTC-USD",
    "confidence": 0.0-1.0,
    "size_pct": 10-35,
    "entry_price": float,
    "stop_loss": float,
    "take_profit": float,
    "strategy": "strategy_name",
    "reasoning": "Human readable reason",
}
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

log = logging.getLogger("strategies")


# ─── STRATEGY 1: LIQUIDATION CASCADE REVERSAL ────────────────────
# Highest alpha. When massive liquidations happen (OI drops + volume
# spikes + price crashes), buy the reversal. From v1's brain.py.

def strategy_liquidation_cascade(pair, indicators_1h, onchain_data, tv_analysis):
    """Buy liquidation cascade reversals — highest alpha strategy."""
    price = indicators_1h.get("price", 0)
    atr = indicators_1h.get("atr", 0) or (price * 0.025)
    atr_pct = indicators_1h.get("atr_pct", 2.0) or 2.0
    vol_ratio = indicators_1h.get("vol_ratio", 1.0) or 1.0
    change_24h = indicators_1h.get("24h_change_pct")
    high_24h = indicators_1h.get("high_24h", price)
    low_24h = indicators_1h.get("low_24h", price)

    if not price or not atr:
        return None

    # Check OI data for cascade signal
    oi = onchain_data.get("open_interest", {})
    oi_change_4h = oi.get("change_4h_pct", 0) or oi.get("change_2h_pct", 0) or 0

    # Detect cascade: rapid OI drop (>5% in 4h) + volume spike (>2.5x) + price drop (>2x ATR)
    price_drop_pct = abs(change_24h) if change_24h and change_24h < 0 else 0
    oi_dropping = oi_change_4h < -5 if oi_change_4h else False

    # If we have OI data, use it. If not, use price action proxy.
    if oi_dropping and vol_ratio > 2.5 and price_drop_pct > atr_pct * 2:
        cascade_detected = True
        cascade_source = "OI"
    elif vol_ratio > 2.5 and price_drop_pct > atr_pct * 2:
        cascade_detected = True
        cascade_source = "price_action"
    else:
        cascade_detected = False
        cascade_source = None

    if not cascade_detected:
        return None

    # Reversal confirmation: look for bullish candle close
    # We check if current price > open (approximated by checking if price recovered)
    # Using 24h high/low to estimate recovery
    if price <= low_24h * 1.005:
        # Price is still at the bottom, no reversal yet
        return None

    # Recovery from the low — this is the reversal signal
    recovery_pct = (price - low_24h) / low_24h * 100 if low_24h > 0 else 0
    if recovery_pct < 0.3:
        return None  # Not enough recovery

    # Confidence scales with cascade size
    confidence = 0.75
    if oi_dropping:
        confidence += 0.05
        if oi_change_4h < -10:
            confidence += 0.05
    if vol_ratio > 4.0:
        confidence += 0.05
    if price_drop_pct > atr_pct * 3:
        confidence += 0.05
    if recovery_pct > 1.0:
        confidence += 0.05  # Strong bounce = stronger signal

    confidence = min(confidence, 0.95)

    # Calculate VWAP target if we have high/low data
    vwap_target = indicators_1h.get("vwap", price * 1.03)
    if vwap_target and vwap_target > price:
        take_profit = round(vwap_target, 2)
    else:
        take_profit = round(price + atr * 3.0, 2)

    stop_loss = round(low_24h - atr * 0.5, 2)

    return {
        "action": "buy",
        "pair": pair,
        "confidence": confidence,
        "size_pct": 30,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "strategy": "liquidation_cascade",
        "reasoning": (
            f"CASCADE REVERSAL [{cascade_source}]: "
            f"OI {oi_change_4h:+.1f}% vol {vol_ratio:.1f}x "
            f"drop {price_drop_pct:.1f}% recovery {recovery_pct:.1f}% "
            f"target ${take_profit:,.0f}"
        ),
    }


# ─── STRATEGY 2: SMART MONEY DIVERGENCE ──────────────────────────
# When large traders are positioned opposite to the crowd,
# follow the smart money via OKX long/short ratio data.

def strategy_smart_money_divergence(pair, indicators_1h, onchain_data, tv_analysis):
    """Follow smart money when they diverge from retail."""
    price = indicators_1h.get("price", 0)
    atr = indicators_1h.get("atr", 0) or (price * 0.025)

    if not price or not atr:
        return None

    # OKX top trader L/S ratio vs global L/S ratio
    top_traders = onchain_data.get("top_traders", {})
    long_short = onchain_data.get("long_short_ratio", {})

    top_ls = top_traders.get("ratio", None)
    global_ls = long_short.get("current", None)

    if top_ls is None or global_ls is None:
        return None

    # Divergence: smart money (top traders) disagrees with crowd
    # top_ls > 1 = big traders are net long
    # global_ls < 1 = crowd is net short
    divergence = top_ls - global_ls

    if abs(divergence) < 0.3:
        return None  # Not enough divergence

    confidence = 0.60
    rsi = indicators_1h.get("rsi", 50)

    if divergence > 0.3:
        # Smart money is MORE LONG than crowd = BUY
        if divergence > 0.6:
            confidence += 0.10
        if divergence > 1.0:
            confidence += 0.10
        if top_traders.get("whales_long"):
            confidence += 0.05
        if rsi and rsi < 50:
            confidence += 0.05  # Buying when RSI isn't overbought

        confidence = min(confidence, 0.85)

        stop = round(price - atr * 2.0, 2)
        target = round(price + atr * 4.0, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "strategy": "smart_money_divergence",
            "reasoning": (
                f"SMART MONEY BUY: top L/S={top_ls:.2f} vs global L/S={global_ls:.2f} "
                f"divergence={divergence:+.2f} RSI={rsi:.0f}"
            ),
        }

    elif divergence < -0.3:
        # Smart money is MORE SHORT than crowd = SELL
        confidence += 0.05 if divergence < -0.6 else 0
        confidence += 0.05 if divergence < -1.0 else 0
        confidence = min(confidence, 0.85)

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": round(price + atr * 2.0, 2),
            "take_profit": round(price - atr * 4.0, 2),
            "strategy": "smart_money_divergence",
            "reasoning": (
                f"SMART MONEY SELL: top L/S={top_ls:.2f} vs global L/S={global_ls:.2f} "
                f"divergence={divergence:+.2f}"
            ),
        }

    return None


# ─── STRATEGY 3: FUNDING RATE FADE ───────────────────────────────
# Extreme funding rates predict mean reversion.
# Negative funding = shorts are paying = squeeze setup.

def strategy_funding_rate_fade(pair, indicators_1h, onchain_data, tv_analysis):
    """Trade against extreme funding rates — #1 edge in crypto."""
    price = indicators_1h.get("price", 0)
    atr = indicators_1h.get("atr", 0) or (price * 0.025)
    rsi = indicators_1h.get("rsi", 50)

    if not price or not atr:
        return None

    funding = onchain_data.get("funding", {})
    current_rate = funding.get("current", None)

    if current_rate is None:
        return None

    # Check signal from v1's pre-processed data if available
    signal = funding.get("signal", {})
    neg_streak = funding.get("negative_streak", 0)
    pos_streak = funding.get("positive_streak", 0)

    # EXTREME NEGATIVE (<-0.01%): BUY signal — short squeeze incoming
    if current_rate < -0.01 or (signal.get("bias") == "bullish" and signal.get("strength") == "strong"):
        confidence = 0.55

        # Scale confidence by how negative
        if current_rate < -0.03:
            confidence += 0.15
        elif current_rate < -0.02:
            confidence += 0.10
        elif current_rate < -0.01:
            confidence += 0.05

        # Streak boost
        if neg_streak >= 3:
            confidence += 0.05
        if neg_streak >= 6:
            confidence += 0.05

        # RSI confirmation (not overbought)
        if rsi and rsi < 60:
            confidence += 0.05

        # Taker data boost
        taker = onchain_data.get("taker_ratio", {})
        if taker.get("aggressive_buyers"):
            confidence += 0.05

        confidence = min(confidence, 0.80)

        stop = round(price - atr * 2.0, 2)
        target = round(price + atr * 4.0, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "strategy": "funding_rate_fade",
            "reasoning": (
                f"SHORT SQUEEZE: FR={current_rate:.4f}% neg_streak={neg_streak} "
                f"RSI={rsi:.0f}"
            ),
        }

    # EXTREME POSITIVE (>0.03%): SELL signal — longs overleveraged
    if current_rate > 0.03 or (signal.get("bias") == "bearish" and signal.get("strength") == "strong"):
        confidence = 0.55

        if current_rate > 0.06:
            confidence += 0.15
        elif current_rate > 0.04:
            confidence += 0.10
        elif current_rate > 0.03:
            confidence += 0.05

        if pos_streak >= 3:
            confidence += 0.05

        if rsi and rsi > 60:
            confidence += 0.05

        confidence = min(confidence, 0.80)

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": round(price + atr * 2.0, 2),
            "take_profit": round(price - atr * 4.0, 2),
            "strategy": "funding_rate_fade",
            "reasoning": (
                f"LONG SQUEEZE: FR={current_rate:.4f}% pos_streak={pos_streak} "
                f"RSI={rsi:.0f}"
            ),
        }

    return None


# ─── STRATEGY 4: TRADINGVIEW CONSENSUS ────────────────────────────
# When TradingView's aggregate of 20+ indicators says STRONG_BUY
# or STRONG_SELL, follow it.

def strategy_tradingview_consensus(pair, indicators_1h, onchain_data, tv_analysis):
    """Trade TradingView's consensus signals."""
    if not tv_analysis:
        return None

    price = indicators_1h.get("price", 0)
    atr = indicators_1h.get("atr", 0) or (price * 0.025)

    if not price or not atr:
        return None

    recommendation = tv_analysis.get("RECOMMENDATION", "NEUTRAL")
    buy_count = tv_analysis.get("BUY", 0)
    sell_count = tv_analysis.get("SELL", 0)
    neutral_count = tv_analysis.get("NEUTRAL", 0)

    # STRONG_BUY: BUY count > 15
    if buy_count > 15 or recommendation == "STRONG_BUY":
        confidence = 0.70
        if buy_count > 18:
            confidence += 0.05
        if buy_count > 20:
            confidence += 0.05

        stop = round(price - atr * 2.0, 2)
        target = round(price + atr * 3.0, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": min(confidence, 0.80),
            "size_pct": 20,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "strategy": "tv_consensus",
            "reasoning": (
                f"TV STRONG_BUY: {buy_count}B/{sell_count}S/{neutral_count}N "
                f"rec={recommendation}"
            ),
        }

    # BUY: BUY count > 10
    if buy_count > 10 or recommendation == "BUY":
        confidence = 0.55
        if buy_count > 13:
            confidence += 0.05

        stop = round(price - atr * 2.0, 2)
        target = round(price + atr * 3.0, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": min(confidence, 0.65),
            "size_pct": 20,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "strategy": "tv_consensus",
            "reasoning": (
                f"TV BUY: {buy_count}B/{sell_count}S/{neutral_count}N "
                f"rec={recommendation}"
            ),
        }

    # STRONG_SELL: SELL count > 15
    if sell_count > 15 or recommendation == "STRONG_SELL":
        confidence = 0.65
        if sell_count > 18:
            confidence += 0.05

        return {
            "action": "sell",
            "pair": pair,
            "confidence": min(confidence, 0.75),
            "size_pct": 20,
            "entry_price": price,
            "stop_loss": round(price + atr * 2.0, 2),
            "take_profit": round(price - atr * 3.0, 2),
            "strategy": "tv_consensus",
            "reasoning": (
                f"TV STRONG_SELL: {buy_count}B/{sell_count}S/{neutral_count}N "
                f"rec={recommendation}"
            ),
        }

    return None


# ─── STRATEGY 5: CVD DIVERGENCE ──────────────────────────────────
# Cumulative Volume Delta divergence. When price drops but buying
# flow increases (accumulation), buy.

def strategy_cvd_divergence(pair, indicators_1h, onchain_data, tv_analysis):
    """Trade CVD/price divergences — accumulation/distribution detection."""
    price = indicators_1h.get("price", 0)
    atr = indicators_1h.get("atr", 0) or (price * 0.025)
    change_24h = indicators_1h.get("24h_change_pct")

    if not price or not atr or change_24h is None:
        return None

    taker_buy = onchain_data.get("taker_buy_vol")
    taker_sell = onchain_data.get("taker_sell_vol")

    if taker_buy is None or taker_sell is None:
        # Try alternative data format
        taker = onchain_data.get("taker_ratio", {})
        buy_vol = taker.get("buy_vol")
        sell_vol = taker.get("sell_vol")
        if buy_vol and sell_vol:
            taker_buy = buy_vol
            taker_sell = sell_vol
        else:
            return None

    # Build CVD: net buying pressure
    cvd = taker_buy - taker_sell
    cvd_ratio = taker_buy / taker_sell if taker_sell > 0 else 1.0

    # ACCUMULATION: Price down + CVD positive (net buying)
    if change_24h < -1.5 and cvd > 0 and cvd_ratio > 1.05:
        confidence = 0.55

        if cvd_ratio > 1.15:
            confidence += 0.10
        elif cvd_ratio > 1.10:
            confidence += 0.05

        if change_24h < -3:
            confidence += 0.05  # Deeper drop = stronger signal

        rsi = indicators_1h.get("rsi", 50)
        if rsi and rsi < 40:
            confidence += 0.05

        confidence = min(confidence, 0.75)

        stop = round(price - atr * 2.0, 2)
        target = round(price + atr * 3.5, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 20,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "strategy": "cvd_divergence",
            "reasoning": (
                f"ACCUMULATION: price {change_24h:+.1f}% but CVD ratio={cvd_ratio:.2f} "
                f"(net buy) RSI={rsi:.0f}"
            ),
        }

    # DISTRIBUTION: Price up + CVD negative (net selling)
    if change_24h > 1.5 and cvd < 0 and cvd_ratio < 0.95:
        confidence = 0.55

        if cvd_ratio < 0.85:
            confidence += 0.10
        elif cvd_ratio < 0.90:
            confidence += 0.05

        if change_24h > 3:
            confidence += 0.05

        confidence = min(confidence, 0.75)

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 20,
            "entry_price": price,
            "stop_loss": round(price + atr * 2.0, 2),
            "take_profit": round(price - atr * 3.5, 2),
            "strategy": "cvd_divergence",
            "reasoning": (
                f"DISTRIBUTION: price {change_24h:+.1f}% but CVD ratio={cvd_ratio:.2f} "
                f"(net sell)"
            ),
        }

    return None


# ─── STRATEGY 6: MOMENTUM BREAKOUT ───────────────────────────────
# When price breaks above 24h high with volume confirmation,
# ride the momentum.

def strategy_momentum_breakout(pair, indicators_1h, onchain_data, tv_analysis):
    """Trade momentum breakouts with volume confirmation."""
    price = indicators_1h.get("price", 0)
    atr = indicators_1h.get("atr", 0) or (price * 0.025)
    high_24h = indicators_1h.get("high_24h")
    low_24h = indicators_1h.get("low_24h")
    vol_ratio = indicators_1h.get("vol_ratio", 1.0) or 1.0
    adx = indicators_1h.get("adx", 0) or 0

    if not price or not atr or not high_24h or not low_24h:
        return None

    # BULLISH BREAKOUT: price > 24h high + volume + ADX
    if price > high_24h and vol_ratio > 1.5 and adx > 20:
        confidence = 0.50

        # How far above the breakout?
        breakout_pct = (price - high_24h) / high_24h * 100
        if breakout_pct > 1.0:
            confidence += 0.05
        if breakout_pct > 2.0:
            confidence += 0.05

        # Volume confirmation
        if vol_ratio > 2.0:
            confidence += 0.05
        if vol_ratio > 3.0:
            confidence += 0.05

        # ADX trend strength
        if adx > 30:
            confidence += 0.05
        if adx > 40:
            confidence += 0.05

        # DI confirmation
        di_plus = indicators_1h.get("di_plus", 0) or 0
        di_minus = indicators_1h.get("di_minus", 0) or 0
        if di_plus > di_minus:
            confidence += 0.05

        confidence = min(confidence, 0.70)

        # Stop below the breakout level
        stop = round(high_24h - atr * 0.5, 2)
        # Target = 2x ATR above entry
        target = round(price + atr * 2.0, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 20,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "strategy": "momentum_breakout",
            "reasoning": (
                f"BREAKOUT: price ${price:,.0f} > 24h high ${high_24h:,.0f} "
                f"(+{breakout_pct:.1f}%) vol={vol_ratio:.1f}x ADX={adx:.0f}"
            ),
        }

    # BEARISH BREAKDOWN: price < 24h low + volume + ADX
    if price < low_24h and vol_ratio > 1.5 and adx > 20:
        confidence = 0.50

        breakdown_pct = (low_24h - price) / low_24h * 100
        if breakdown_pct > 1.0:
            confidence += 0.05
        if breakdown_pct > 2.0:
            confidence += 0.05
        if vol_ratio > 2.0:
            confidence += 0.05
        if adx > 30:
            confidence += 0.05

        di_plus = indicators_1h.get("di_plus", 0) or 0
        di_minus = indicators_1h.get("di_minus", 0) or 0
        if di_minus > di_plus:
            confidence += 0.05

        confidence = min(confidence, 0.70)

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 20,
            "entry_price": price,
            "stop_loss": round(low_24h + atr * 0.5, 2),
            "take_profit": round(price - atr * 2.0, 2),
            "strategy": "momentum_breakout",
            "reasoning": (
                f"BREAKDOWN: price ${price:,.0f} < 24h low ${low_24h:,.0f} "
                f"(-{breakdown_pct:.1f}%) vol={vol_ratio:.1f}x ADX={adx:.0f}"
            ),
        }

    return None


# ─── STRATEGY 7: MEAN REVERSION EXTREME ──────────────────────────
# Aggressive mean reversion at Bollinger Band extremes.

def strategy_mean_reversion_extreme(pair, indicators_1h, onchain_data, tv_analysis):
    """Aggressive mean reversion at BB extremes + RSI + StochRSI confluence."""
    price = indicators_1h.get("price", 0)
    atr = indicators_1h.get("atr", 0) or (price * 0.025)
    bb_pct = indicators_1h.get("bb_pct")
    rsi = indicators_1h.get("rsi")
    stoch_k = indicators_1h.get("stoch_rsi_k")
    bb_middle = indicators_1h.get("bb_middle")

    if not all(v is not None for v in [price, atr, bb_pct, rsi, stoch_k]):
        return None

    # STRONG BUY: BB% < 0.05 + RSI < 25 + StochRSI < 0.10
    if bb_pct < 0.05 and rsi < 25 and stoch_k < 0.10:
        confidence = 0.60

        # Extreme oversold boosts
        if bb_pct < 0.0:
            confidence += 0.08
        if rsi < 20:
            confidence += 0.07
        if stoch_k < 0.05:
            confidence += 0.05
        if bb_pct < -0.05:
            confidence += 0.05  # Way below lower band

        confidence = min(confidence, 0.85)

        target = bb_middle if bb_middle else price * 1.04
        stop = round(price - atr * 1.5, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": round(target, 2),
            "strategy": "mean_reversion_extreme",
            "reasoning": (
                f"EXTREME OVERSOLD: BB%={bb_pct:.3f} RSI={rsi:.0f} "
                f"StochK={stoch_k:.2f} target=${target:,.0f}"
            ),
        }

    # STRONG SELL: BB% > 0.95 + RSI > 75 + StochRSI > 0.90
    if bb_pct > 0.95 and rsi > 75 and stoch_k > 0.90:
        confidence = 0.60

        if bb_pct > 1.0:
            confidence += 0.08
        if rsi > 80:
            confidence += 0.07
        if stoch_k > 0.95:
            confidence += 0.05
        if bb_pct > 1.05:
            confidence += 0.05

        confidence = min(confidence, 0.85)

        target = bb_middle if bb_middle else price * 0.96

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": round(price + atr * 1.5, 2),
            "take_profit": round(target, 2),
            "strategy": "mean_reversion_extreme",
            "reasoning": (
                f"EXTREME OVERBOUGHT: BB%={bb_pct:.3f} RSI={rsi:.0f} "
                f"StochK={stoch_k:.2f} target=${target:,.0f}"
            ),
        }

    return None


# ─── MAIN ANALYZER ────────────────────────────────────────────────

ALL_STRATEGIES = [
    strategy_liquidation_cascade,
    strategy_smart_money_divergence,
    strategy_funding_rate_fade,
    strategy_tradingview_consensus,
    strategy_cvd_divergence,
    strategy_momentum_breakout,
    strategy_mean_reversion_extreme,
]


def analyze(pair, indicators_1h, onchain_data=None, tv_analysis=None):
    """Run ALL strategies in parallel, return the best signal.

    Args:
        pair: Trading pair string (e.g. "BTC-USD")
        indicators_1h: Dict from indicators.compute_all()
        onchain_data: Dict with OKX/Binance derivatives data
        tv_analysis: Dict with TradingView analysis data

    Returns:
        Best signal dict (highest confidence) or None
    """
    if not indicators_1h:
        return None

    onchain_data = onchain_data or {}
    tv_analysis = tv_analysis or {}

    signals = []

    # Run all strategies in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(ALL_STRATEGIES)) as executor:
        futures = {}
        for strat_fn in ALL_STRATEGIES:
            future = executor.submit(strat_fn, pair, indicators_1h, onchain_data, tv_analysis)
            futures[future] = strat_fn.__name__

        for future in as_completed(futures):
            strat_name = futures[future]
            try:
                result = future.result(timeout=5)
                if result is not None:
                    # Ensure pair is set
                    result["pair"] = pair
                    signals.append(result)
                    log.info(f"  [{strat_name}] {result['action'].upper()} "
                             f"conf={result['confidence']:.2f}: {result['reasoning'][:80]}")
            except Exception as e:
                log.warning(f"  [{strat_name}] error: {e}")

    if not signals:
        return None

    # Sort by confidence, highest first
    signals.sort(key=lambda s: s["confidence"], reverse=True)

    best = signals[0]
    log.info(f"  BEST: {best['strategy']} {best['action'].upper()} "
             f"conf={best['confidence']:.2f}")

    return best
