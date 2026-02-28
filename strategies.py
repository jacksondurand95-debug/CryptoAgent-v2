"""CryptoAgent v2.1 — Proven Algorithm Trading Strategies.

Based on:
1. AdaptiveTrend (arXiv 2602.11708): Sharpe 2.41, 6H momentum + dynamic trailing
2. Bollinger-Keltner Squeeze Breakout: fewer trades, bigger moves
3. Funding Rate Fade: highest edge in crypto derivatives
4. Smart Money Divergence: follow whales, fade retail
5. CVD Divergence: accumulation/distribution detection
6. Liquidation Cascade Reversal: buy capitulation events
7. Time-Series Momentum: 28-period lookback, multi-day holds

ALL strategies target 2%+ moves to clear 0.80% round-trip fees.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

log = logging.getLogger("strategies")


# ─── STRATEGY 1: ADAPTIVE TREND MOMENTUM (Sharpe 2.41) ─────────
# From arXiv 2602.11708v1. 6H timeframe, dynamic trailing stops,
# momentum-based entries. The gold standard.

def strategy_adaptive_trend(pair, ind_6h, ind_1d, onchain_data, tv_analysis):
    """AdaptiveTrend: momentum entry + dynamic trailing stop.

    Entry: 6H momentum > threshold + daily trend alignment + volume.
    Exit: Dynamic trailing stop at 2.5x ATR from peak.
    """
    price = ind_6h.get("price", 0)
    atr = ind_6h.get("atr", 0) or (price * 0.025)

    if not price or not atr:
        return None

    # Momentum signal (core of AdaptiveTrend)
    mom_28 = ind_6h.get("momentum_28", 0)
    mom_14 = ind_6h.get("momentum_14", 0)
    mom_7 = ind_6h.get("momentum_7", 0)
    rolling_sharpe = ind_6h.get("rolling_sharpe", 0)

    # Daily trend filter (only trade with the trend)
    daily_trend = ind_1d.get("ema_trend", "mixed") if ind_1d else "mixed"
    daily_ema9 = ind_1d.get("ema_9", 0) if ind_1d else 0
    daily_ema21 = ind_1d.get("ema_21", 0) if ind_1d else 0

    # 6H EMA alignment
    ema_9 = ind_6h.get("ema_9", 0)
    ema_21 = ind_6h.get("ema_21", 0)
    ema_trend_6h = ind_6h.get("ema_trend", "mixed")
    vol_ratio = ind_6h.get("vol_ratio", 1.0) or 1.0

    # ── LONG ENTRY ──
    # Conditions: positive momentum + daily uptrend + 6H EMA alignment
    if (mom_28 > config.MOMENTUM_THRESHOLD
            and mom_14 > 0
            and daily_trend in ("bullish", "mixed")
            and (daily_ema9 > daily_ema21 or daily_trend == "mixed")):

        confidence = 0.65

        # Momentum strength
        if mom_28 > 0.05:
            confidence += 0.05
        if mom_28 > 0.10:
            confidence += 0.05

        # Sharpe filter (paper uses >= 1.3 for longs)
        if rolling_sharpe >= config.SHARPE_FILTER_LONG:
            confidence += 0.05
        elif rolling_sharpe < 0.5:
            confidence -= 0.10  # Poor risk-adjusted returns

        # EMA alignment boost
        if ema_trend_6h == "bullish":
            confidence += 0.05

        # Volume confirmation
        if vol_ratio > 1.5:
            confidence += 0.03

        # MACD confirmation
        if ind_6h.get("macd_hist", 0) and ind_6h["macd_hist"] > 0:
            confidence += 0.03

        confidence = max(0.40, min(confidence, 0.90))

        # Dynamic stops from the paper: 2.5x ATR
        stop = round(price - config.STOP_LOSS_ATR_MULT * atr, 2)
        target = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "atr": atr,
            "strategy": "adaptive_trend",
            "reasoning": (
                f"ADAPTIVE TREND: mom28={mom_28:+.3f} mom14={mom_14:+.3f} "
                f"sharpe={rolling_sharpe:.1f} daily={daily_trend} "
                f"vol={vol_ratio:.1f}x"
            ),
        }

    # ── SELL signal when momentum turns negative on open position ──
    if mom_14 < -config.MOMENTUM_THRESHOLD and mom_7 < 0:
        confidence = 0.60
        if mom_28 < 0:
            confidence += 0.05
        if ema_trend_6h == "bearish":
            confidence += 0.05

        return {
            "action": "sell",
            "pair": pair,
            "confidence": min(confidence, 0.80),
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": round(price + config.STOP_LOSS_ATR_MULT * atr, 2),
            "take_profit": round(price - config.TAKE_PROFIT_ATR_MULT * atr, 2),
            "strategy": "adaptive_trend",
            "reasoning": (
                f"TREND REVERSAL: mom14={mom_14:+.3f} mom7={mom_7:+.3f} "
                f"EMA={ema_trend_6h}"
            ),
        }

    return None


# ─── STRATEGY 2: MOMENTUM SQUEEZE BREAKOUT ──────────────────────
# Bollinger Bands inside Keltner Channel = volatility squeeze.
# When squeeze releases with volume, ride the breakout.

def strategy_momentum_squeeze(pair, ind_6h, ind_1d, onchain_data, tv_analysis):
    """BB-KC squeeze breakout — fewer trades, bigger moves."""
    price = ind_6h.get("price", 0)
    atr = ind_6h.get("atr", 0) or (price * 0.025)

    if not price or not atr:
        return None

    squeeze_bars = ind_6h.get("squeeze_bars", 0)
    squeeze_releasing = ind_6h.get("squeeze_releasing", False)
    kc_upper = ind_6h.get("kc_upper")
    kc_lower = ind_6h.get("kc_lower")
    bb_upper = ind_6h.get("bb_upper")
    vol_ratio = ind_6h.get("vol_ratio", 1.0) or 1.0

    if not kc_upper or not kc_lower:
        return None

    # ── SQUEEZE BREAKOUT (just released or breaking now) ──
    # Squeeze must have been active for at least 3 bars
    if (squeeze_releasing or squeeze_bars >= config.SQUEEZE_MIN_BARS) and vol_ratio > 1.3:

        # Determine breakout direction
        if price > kc_upper:
            # BULLISH breakout above Keltner upper
            confidence = 0.70

            # How strong is the breakout?
            breakout_pct = (price - kc_upper) / kc_upper * 100
            if breakout_pct > 0.5:
                confidence += 0.05
            if breakout_pct > 1.0:
                confidence += 0.05

            # Volume confirms
            if vol_ratio > 2.0:
                confidence += 0.05
            if vol_ratio > 3.0:
                confidence += 0.03

            # Squeeze duration (longer = more explosive)
            if squeeze_bars >= 6:
                confidence += 0.05
            if squeeze_bars >= 10:
                confidence += 0.03

            # Daily trend alignment
            daily_trend = ind_1d.get("ema_trend", "mixed") if ind_1d else "mixed"
            if daily_trend == "bullish":
                confidence += 0.05

            # Funding rate not overbought
            funding = onchain_data.get("funding", {})
            fr = funding.get("current", 0)
            if fr is not None and fr < 0.03:
                confidence += 0.03

            confidence = min(confidence, 0.90)

            stop = round(price - config.STOP_LOSS_ATR_MULT * atr, 2)
            target = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 2)

            return {
                "action": "buy",
                "pair": pair,
                "confidence": confidence,
                "size_pct": 25,
                "entry_price": price,
                "stop_loss": stop,
                "take_profit": target,
                "atr": atr,
                "strategy": "momentum_squeeze",
                "reasoning": (
                    f"SQUEEZE BREAKOUT: {squeeze_bars} bars squeezed, "
                    f"price>${kc_upper:,.0f} KC_upper, vol={vol_ratio:.1f}x "
                    f"breakout={breakout_pct:+.1f}%"
                ),
            }

        elif price < kc_lower:
            # BEARISH breakdown below Keltner lower
            confidence = 0.65

            breakdown_pct = (kc_lower - price) / kc_lower * 100
            if breakdown_pct > 0.5:
                confidence += 0.05
            if vol_ratio > 2.0:
                confidence += 0.05
            if squeeze_bars >= 6:
                confidence += 0.05

            confidence = min(confidence, 0.80)

            return {
                "action": "sell",
                "pair": pair,
                "confidence": confidence,
                "size_pct": 25,
                "entry_price": price,
                "stop_loss": round(price + config.STOP_LOSS_ATR_MULT * atr, 2),
                "take_profit": round(price - config.TAKE_PROFIT_ATR_MULT * atr, 2),
                "atr": atr,
                "strategy": "momentum_squeeze",
                "reasoning": (
                    f"SQUEEZE BREAKDOWN: {squeeze_bars} bars, "
                    f"price<${kc_lower:,.0f} KC_lower, vol={vol_ratio:.1f}x"
                ),
            }

    return None


# ─── STRATEGY 3: FUNDING RATE FADE ──────────────────────────────
# Extreme funding rates predict mean reversion.
# Negative = shorts paying = squeeze setup.

def strategy_funding_rate_fade(pair, ind_6h, ind_1d, onchain_data, tv_analysis):
    """Trade against extreme funding rates — proven crypto edge."""
    price = ind_6h.get("price", 0)
    atr = ind_6h.get("atr", 0) or (price * 0.025)
    rsi = ind_6h.get("rsi", 50)

    if not price or not atr:
        return None

    funding = onchain_data.get("funding", {})
    current_rate = funding.get("current", None)

    if current_rate is None:
        return None

    signal = funding.get("signal", {})
    neg_streak = funding.get("negative_streak", 0)
    pos_streak = funding.get("positive_streak", 0)

    # EXTREME NEGATIVE (<-0.02%): BUY — short squeeze incoming
    if current_rate < -0.02 or (signal.get("bias") == "bullish" and signal.get("strength") == "strong"):
        confidence = 0.65

        if current_rate < -0.05:
            confidence += 0.10
        elif current_rate < -0.03:
            confidence += 0.05

        if neg_streak >= 3:
            confidence += 0.05

        if rsi and rsi < 50:
            confidence += 0.03

        # Taker data boost
        taker = onchain_data.get("taker_ratio", {})
        if taker.get("aggressive_buyers"):
            confidence += 0.03

        # Daily trend alignment
        daily_trend = ind_1d.get("ema_trend", "mixed") if ind_1d else "mixed"
        if daily_trend == "bullish":
            confidence += 0.05

        confidence = min(confidence, 0.85)

        stop = round(price - config.STOP_LOSS_ATR_MULT * atr, 2)
        target = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "atr": atr,
            "strategy": "funding_rate_fade",
            "reasoning": (
                f"SHORT SQUEEZE: FR={current_rate:.4f}% neg_streak={neg_streak} "
                f"RSI={rsi:.0f} daily={daily_trend if ind_1d else 'N/A'}"
            ),
        }

    # EXTREME POSITIVE (>0.05%): SELL — longs overleveraged
    if current_rate > 0.05 or (signal.get("bias") == "bearish" and signal.get("strength") == "strong"):
        confidence = 0.65

        if current_rate > 0.08:
            confidence += 0.10
        elif current_rate > 0.06:
            confidence += 0.05

        if pos_streak >= 3:
            confidence += 0.05
        if rsi and rsi > 60:
            confidence += 0.03

        confidence = min(confidence, 0.85)

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": round(price + config.STOP_LOSS_ATR_MULT * atr, 2),
            "take_profit": round(price - config.TAKE_PROFIT_ATR_MULT * atr, 2),
            "atr": atr,
            "strategy": "funding_rate_fade",
            "reasoning": (
                f"LONG SQUEEZE: FR={current_rate:.4f}% pos_streak={pos_streak} "
                f"RSI={rsi:.0f}"
            ),
        }

    return None


# ─── STRATEGY 4: SMART MONEY DIVERGENCE ──────────────────────────
# Follow smart money (top traders) when they diverge from retail.

def strategy_smart_money_divergence(pair, ind_6h, ind_1d, onchain_data, tv_analysis):
    """Follow smart money when they diverge from retail."""
    price = ind_6h.get("price", 0)
    atr = ind_6h.get("atr", 0) or (price * 0.025)

    if not price or not atr:
        return None

    top_traders = onchain_data.get("top_traders", {})
    long_short = onchain_data.get("long_short_ratio", {})

    top_ls = top_traders.get("ratio", None)
    global_ls = long_short.get("current", None)

    if top_ls is None or global_ls is None:
        return None

    divergence = top_ls - global_ls

    if abs(divergence) < 0.4:
        return None  # Need strong divergence (raised from 0.3)

    rsi = ind_6h.get("rsi", 50)

    if divergence > 0.4:
        # Smart money MORE LONG than crowd = BUY
        confidence = 0.65

        if divergence > 0.8:
            confidence += 0.10
        elif divergence > 0.6:
            confidence += 0.05

        if top_traders.get("whales_long"):
            confidence += 0.05

        # Daily trend alignment
        daily_trend = ind_1d.get("ema_trend", "mixed") if ind_1d else "mixed"
        if daily_trend == "bullish":
            confidence += 0.05

        confidence = min(confidence, 0.85)

        stop = round(price - config.STOP_LOSS_ATR_MULT * atr, 2)
        target = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "atr": atr,
            "strategy": "smart_money_divergence",
            "reasoning": (
                f"SMART MONEY BUY: top={top_ls:.2f} vs retail={global_ls:.2f} "
                f"div={divergence:+.2f} RSI={rsi:.0f}"
            ),
        }

    elif divergence < -0.4:
        confidence = 0.65
        if divergence < -0.8:
            confidence += 0.10
        elif divergence < -0.6:
            confidence += 0.05

        confidence = min(confidence, 0.85)

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": round(price + config.STOP_LOSS_ATR_MULT * atr, 2),
            "take_profit": round(price - config.TAKE_PROFIT_ATR_MULT * atr, 2),
            "atr": atr,
            "strategy": "smart_money_divergence",
            "reasoning": (
                f"SMART MONEY SELL: top={top_ls:.2f} vs retail={global_ls:.2f} "
                f"div={divergence:+.2f}"
            ),
        }

    return None


# ─── STRATEGY 5: CVD DIVERGENCE ──────────────────────────────────
# Cumulative Volume Delta divergence — accumulation/distribution.

def strategy_cvd_divergence(pair, ind_6h, ind_1d, onchain_data, tv_analysis):
    """Trade CVD/price divergences — accumulation/distribution."""
    price = ind_6h.get("price", 0)
    atr = ind_6h.get("atr", 0) or (price * 0.025)
    change_pct = ind_6h.get("24h_change_pct")  # Will be 4-candle change for 6H

    if not price or not atr or change_pct is None:
        return None

    taker_buy = onchain_data.get("taker_buy_vol")
    taker_sell = onchain_data.get("taker_sell_vol")

    if taker_buy is None or taker_sell is None:
        taker = onchain_data.get("taker_ratio", {})
        buy_vol = taker.get("buy_vol")
        sell_vol = taker.get("sell_vol")
        if buy_vol and sell_vol:
            taker_buy = buy_vol
            taker_sell = sell_vol
        else:
            return None

    cvd_ratio = taker_buy / taker_sell if taker_sell > 0 else 1.0

    # ACCUMULATION: Price down significantly + net buying
    if change_pct < -2.0 and cvd_ratio > 1.10:
        confidence = 0.65

        if cvd_ratio > 1.20:
            confidence += 0.05
        if change_pct < -4:
            confidence += 0.05

        rsi = ind_6h.get("rsi", 50)
        if rsi and rsi < 35:
            confidence += 0.05

        # Daily trend — stronger if buying dip in uptrend
        daily_trend = ind_1d.get("ema_trend", "mixed") if ind_1d else "mixed"
        if daily_trend == "bullish":
            confidence += 0.05

        confidence = min(confidence, 0.80)

        stop = round(price - config.STOP_LOSS_ATR_MULT * atr, 2)
        target = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "atr": atr,
            "strategy": "cvd_divergence",
            "reasoning": (
                f"ACCUMULATION: price {change_pct:+.1f}% but CVD ratio={cvd_ratio:.2f} "
                f"(net buy) RSI={rsi:.0f}"
            ),
        }

    # DISTRIBUTION: Price up significantly + net selling
    if change_pct > 2.0 and cvd_ratio < 0.90:
        confidence = 0.65

        if cvd_ratio < 0.80:
            confidence += 0.05
        if change_pct > 4:
            confidence += 0.05

        confidence = min(confidence, 0.80)

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": round(price + config.STOP_LOSS_ATR_MULT * atr, 2),
            "take_profit": round(price - config.TAKE_PROFIT_ATR_MULT * atr, 2),
            "atr": atr,
            "strategy": "cvd_divergence",
            "reasoning": (
                f"DISTRIBUTION: price {change_pct:+.1f}% but CVD ratio={cvd_ratio:.2f} "
                f"(net sell)"
            ),
        }

    return None


# ─── STRATEGY 6: LIQUIDATION CASCADE REVERSAL ───────────────────
# Buy capitulation events — highest alpha when they happen.

def strategy_liquidation_cascade(pair, ind_6h, ind_1d, onchain_data, tv_analysis):
    """Buy liquidation cascade reversals — highest alpha strategy."""
    price = ind_6h.get("price", 0)
    atr = ind_6h.get("atr", 0) or (price * 0.025)
    atr_pct = ind_6h.get("atr_pct", 2.0) or 2.0
    vol_ratio = ind_6h.get("vol_ratio", 1.0) or 1.0
    change_pct = ind_6h.get("24h_change_pct")
    low_24h = ind_6h.get("low_24h", price)

    if not price or not atr:
        return None

    oi = onchain_data.get("open_interest", {})
    oi_change_4h = oi.get("change_4h_pct", 0) or oi.get("change_2h_pct", 0) or 0

    price_drop_pct = abs(change_pct) if change_pct and change_pct < 0 else 0
    oi_dropping = oi_change_4h < -5 if oi_change_4h else False

    # Cascade: large OI drop + volume spike + big price drop
    if oi_dropping and vol_ratio > 2.0 and price_drop_pct > atr_pct * 1.5:
        cascade_detected = True
    elif vol_ratio > 2.5 and price_drop_pct > atr_pct * 2:
        cascade_detected = True
    else:
        return None

    # Reversal confirmation: price recovered from the low
    if price <= low_24h * 1.005:
        return None

    recovery_pct = (price - low_24h) / low_24h * 100 if low_24h > 0 else 0
    if recovery_pct < 0.5:
        return None

    confidence = 0.75

    if oi_dropping:
        confidence += 0.05
    if vol_ratio > 4.0:
        confidence += 0.05
    if price_drop_pct > atr_pct * 3:
        confidence += 0.05
    if recovery_pct > 1.5:
        confidence += 0.05

    confidence = min(confidence, 0.90)

    vwap_target = ind_6h.get("vwap", price * 1.03)
    if vwap_target and vwap_target > price:
        take_profit = round(vwap_target, 2)
    else:
        take_profit = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 2)

    stop_loss = round(low_24h - atr * 0.5, 2)

    return {
        "action": "buy",
        "pair": pair,
        "confidence": confidence,
        "size_pct": 25,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": atr,
        "strategy": "liquidation_cascade",
        "reasoning": (
            f"CASCADE REVERSAL: OI {oi_change_4h:+.1f}% vol {vol_ratio:.1f}x "
            f"drop {price_drop_pct:.1f}% recovery {recovery_pct:.1f}%"
        ),
    }


# ─── STRATEGY 7: MULTI-TF CONFLUENCE ────────────────────────────
# When 1H, 6H, and 1D all agree on direction with high score.

def strategy_multi_tf_confluence(pair, ind_6h, ind_1d, onchain_data, tv_analysis):
    """Trade when multiple timeframes agree strongly."""
    from indicators import confluence_score

    if not ind_6h or not ind_1d:
        return None

    price = ind_6h.get("price", 0)
    atr = ind_6h.get("atr", 0) or (price * 0.025)

    if not price or not atr:
        return None

    score_6h = confluence_score(ind_6h)
    score_1d = confluence_score(ind_1d)

    # Both timeframes must agree strongly
    if score_6h > 40 and score_1d > 25:
        confidence = 0.65
        combined = (score_6h * 0.6 + score_1d * 0.4)

        if combined > 50:
            confidence += 0.05
        if combined > 60:
            confidence += 0.05

        # TradingView confirmation
        if tv_analysis:
            rec = tv_analysis.get("RECOMMENDATION", "")
            if rec in ("STRONG_BUY", "BUY"):
                confidence += 0.05

        confidence = min(confidence, 0.85)

        stop = round(price - config.STOP_LOSS_ATR_MULT * atr, 2)
        target = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 2)

        return {
            "action": "buy",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": stop,
            "take_profit": target,
            "atr": atr,
            "strategy": "multi_tf_confluence",
            "reasoning": (
                f"MTF BUY: 6H={score_6h:+.0f} 1D={score_1d:+.0f} "
                f"combined={combined:.0f}"
            ),
        }

    elif score_6h < -40 and score_1d < -25:
        confidence = 0.65
        combined = abs(score_6h * 0.6 + score_1d * 0.4)

        if combined > 50:
            confidence += 0.05
        if combined > 60:
            confidence += 0.05

        if tv_analysis:
            rec = tv_analysis.get("RECOMMENDATION", "")
            if rec in ("STRONG_SELL", "SELL"):
                confidence += 0.05

        confidence = min(confidence, 0.85)

        return {
            "action": "sell",
            "pair": pair,
            "confidence": confidence,
            "size_pct": 25,
            "entry_price": price,
            "stop_loss": round(price + config.STOP_LOSS_ATR_MULT * atr, 2),
            "take_profit": round(price - config.TAKE_PROFIT_ATR_MULT * atr, 2),
            "atr": atr,
            "strategy": "multi_tf_confluence",
            "reasoning": (
                f"MTF SELL: 6H={score_6h:+.0f} 1D={score_1d:+.0f} "
                f"combined={combined:.0f}"
            ),
        }

    return None


# ─── MAIN ANALYZER ──────────────────────────────────────────────

ALL_STRATEGIES = [
    strategy_adaptive_trend,
    strategy_momentum_squeeze,
    strategy_funding_rate_fade,
    strategy_smart_money_divergence,
    strategy_cvd_divergence,
    strategy_liquidation_cascade,
    strategy_multi_tf_confluence,
]


def analyze(pair, ind_6h, ind_1d=None, onchain_data=None, tv_analysis=None):
    """Run ALL strategies in parallel, return the best signal.

    Args:
        pair: Trading pair string (e.g. "BTC-USD")
        ind_6h: Dict from indicators.compute_all() on 6H candles (PRIMARY)
        ind_1d: Dict from indicators.compute_all() on 1D candles (TREND FILTER)
        onchain_data: Dict with OKX derivatives data
        tv_analysis: Dict with TradingView analysis data

    Returns:
        Best signal dict (highest confidence above MIN_CONFIDENCE) or None
    """
    if not ind_6h:
        return None

    ind_1d = ind_1d or {}
    onchain_data = onchain_data or {}
    tv_analysis = tv_analysis or {}

    signals = []

    with ThreadPoolExecutor(max_workers=len(ALL_STRATEGIES)) as executor:
        futures = {}
        for strat_fn in ALL_STRATEGIES:
            future = executor.submit(
                strat_fn, pair, ind_6h, ind_1d, onchain_data, tv_analysis
            )
            futures[future] = strat_fn.__name__

        for future in as_completed(futures):
            strat_name = futures[future]
            try:
                result = future.result(timeout=5)
                if result is not None:
                    result["pair"] = pair
                    signals.append(result)
                    log.info(
                        f"  [{strat_name}] {result['action'].upper()} "
                        f"conf={result['confidence']:.2f}: {result['reasoning'][:80]}"
                    )
            except Exception as e:
                log.warning(f"  [{strat_name}] error: {e}")

    if not signals:
        return None

    # Filter by minimum confidence
    signals = [s for s in signals if s["confidence"] >= config.MIN_CONFIDENCE]
    if not signals:
        return None

    # Sort by confidence, highest first
    signals.sort(key=lambda s: s["confidence"], reverse=True)

    best = signals[0]
    log.info(
        f"  BEST: {best['strategy']} {best['action'].upper()} "
        f"conf={best['confidence']:.2f}"
    )

    return best
