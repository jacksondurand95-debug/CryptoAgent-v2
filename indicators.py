"""Technical analysis indicators for trading signals.

Ported from CryptoAgent v1 — standalone, no external config imports.
"""
import pandas as pd
import ta


def compute_all(candles):
    """Compute all indicators from OHLCV candle data.

    Args:
        candles: list of dicts with open/high/low/close/volume keys

    Returns:
        dict with all indicator values for the latest candle
    """
    if len(candles) < 30:
        return {}

    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RSI (14-period)
    rsi = ta.momentum.RSIIndicator(close, window=14)
    df["rsi"] = rsi.rsi()

    # MACD (12, 26, 9)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # EMA 9, 21, 50
    df["ema_9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()  # 0-1 position within bands

    # ATR (14-period)
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr.average_true_range()

    # ADX (14-period)
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()
    df["di_plus"] = adx.adx_pos()
    df["di_minus"] = adx.adx_neg()

    # Volume SMA (20-period)
    df["vol_sma_20"] = volume.rolling(window=20).mean()
    df["vol_ratio"] = volume / df["vol_sma_20"]

    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_k"] = stoch.stochrsi_k()
    df["stoch_rsi_d"] = stoch.stochrsi_d()

    # Get latest values
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    result = {
        "price": round(latest["close"], 2),
        "rsi": round(latest["rsi"], 2) if pd.notna(latest["rsi"]) else None,
        "macd": round(latest["macd"], 4) if pd.notna(latest["macd"]) else None,
        "macd_signal": round(latest["macd_signal"], 4) if pd.notna(latest["macd_signal"]) else None,
        "macd_hist": round(latest["macd_hist"], 4) if pd.notna(latest["macd_hist"]) else None,
        "macd_crossover": (
            "bullish" if (prev["macd"] < prev["macd_signal"] and latest["macd"] > latest["macd_signal"])
            else "bearish" if (prev["macd"] > prev["macd_signal"] and latest["macd"] < latest["macd_signal"])
            else "none"
        ),
        "ema_9": round(latest["ema_9"], 2) if pd.notna(latest["ema_9"]) else None,
        "ema_21": round(latest["ema_21"], 2) if pd.notna(latest["ema_21"]) else None,
        "ema_50": round(latest["ema_50"], 2) if pd.notna(latest["ema_50"]) else None,
        "ema_trend": (
            "bullish" if latest["ema_9"] > latest["ema_21"] > latest["ema_50"]
            else "bearish" if latest["ema_9"] < latest["ema_21"] < latest["ema_50"]
            else "mixed"
        ),
        "bb_upper": round(latest["bb_upper"], 2) if pd.notna(latest["bb_upper"]) else None,
        "bb_lower": round(latest["bb_lower"], 2) if pd.notna(latest["bb_lower"]) else None,
        "bb_pct": round(latest["bb_pct"], 3) if pd.notna(latest["bb_pct"]) else None,
        "atr": round(latest["atr"], 2) if pd.notna(latest["atr"]) else None,
        "atr_pct": round(latest["atr"] / latest["close"] * 100, 2) if pd.notna(latest["atr"]) else None,
        "adx": round(latest["adx"], 2) if pd.notna(latest["adx"]) else None,
        "di_plus": round(latest["di_plus"], 2) if pd.notna(latest["di_plus"]) else None,
        "di_minus": round(latest["di_minus"], 2) if pd.notna(latest["di_minus"]) else None,
        "trend_strength": (
            "strong" if pd.notna(latest["adx"]) and latest["adx"] > 25
            else "weak" if pd.notna(latest["adx"]) and latest["adx"] < 20
            else "moderate"
        ),
        "volume": round(latest["volume"], 2),
        "vol_ratio": round(latest["vol_ratio"], 2) if pd.notna(latest["vol_ratio"]) else None,
        "stoch_rsi_k": round(latest["stoch_rsi_k"], 2) if pd.notna(latest["stoch_rsi_k"]) else None,
        "stoch_rsi_d": round(latest["stoch_rsi_d"], 2) if pd.notna(latest["stoch_rsi_d"]) else None,
        # Price action
        "24h_change_pct": round((latest["close"] - df.iloc[-24]["close"]) / df.iloc[-24]["close"] * 100, 2) if len(df) >= 24 else None,
        "high_24h": round(df["high"].tail(24).max(), 2) if len(df) >= 24 else None,
        "low_24h": round(df["low"].tail(24).min(), 2) if len(df) >= 24 else None,
    }
    # BB middle (for mean reversion targets)
    result["bb_middle"] = round(latest["bb_middle"], 2) if pd.notna(latest["bb_middle"]) else None

    # VWAP (24h rolling volume-weighted average price)
    try:
        lookback = min(24, len(df))
        recent = df.tail(lookback)
        tp = (recent["high"] + recent["low"] + recent["close"]) / 3
        cum_vol = recent["volume"].cumsum()
        cum_tp_vol = (tp * recent["volume"]).cumsum()
        vwap = cum_tp_vol / cum_vol
        vwap_val = float(vwap.iloc[-1])
        vwap_dev = float(close.iloc[-1]) - vwap_val
        vwap_std = float((close.tail(lookback) - vwap).std())
        result["vwap"] = round(vwap_val, 2)
        result["vwap_upper_2"] = round(vwap_val + 2 * vwap_std, 2) if vwap_std else None
        result["vwap_lower_2"] = round(vwap_val - 2 * vwap_std, 2) if vwap_std else None
        result["vwap_deviation_sigma"] = round(vwap_dev / vwap_std, 2) if vwap_std > 0 else 0
    except Exception:
        result["vwap"] = None
        result["vwap_deviation_sigma"] = 0

    return result


# --- Advanced Strategy Signals ---

def confluence_score(indicators):
    """Compute directional confluence score from -100 to +100."""
    score = 0

    ema_trend = indicators.get("ema_trend")
    if ema_trend == "bullish":
        score += 25
    elif ema_trend == "bearish":
        score -= 25

    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi < 30:
            score += 15
        elif rsi < 40:
            score += 8
        elif rsi > 60:
            score -= 8
        elif rsi > 70:
            score -= 15

    macd_hist = indicators.get("macd_hist")
    if macd_hist is not None:
        if macd_hist > 0:
            score += 20
        else:
            score -= 20

    macd_cross = indicators.get("macd_crossover")
    if macd_cross == "bullish":
        score += 10
    elif macd_cross == "bearish":
        score -= 10

    adx = indicators.get("adx")
    if adx is not None:
        if adx > 25:
            score = int(score * 1.15)
        elif adx < 20:
            score = int(score * 0.7)

    vol_ratio = indicators.get("vol_ratio")
    if vol_ratio is not None and vol_ratio > 1.3:
        score += 10 if score > 0 else -10

    bb_pct = indicators.get("bb_pct")
    if bb_pct is not None:
        if bb_pct < 0.1:
            score += 15
        elif bb_pct < 0.2:
            score += 8
        elif bb_pct > 0.8:
            score -= 8
        elif bb_pct > 0.9:
            score -= 15

    stoch_k = indicators.get("stoch_rsi_k")
    if stoch_k is not None:
        if stoch_k < 0.1:
            score += 10
        elif stoch_k > 0.9:
            score -= 10

    return max(-100, min(100, score))


def multi_tf_confluence(candles_1h, candles_6h, candles_1d):
    """Score across 3 timeframes, return direction + confidence + indicators."""
    ind_1h = compute_all(candles_1h) if len(candles_1h) >= 30 else {}
    ind_6h = compute_all(candles_6h) if len(candles_6h) >= 30 else {}
    ind_1d = compute_all(candles_1d) if len(candles_1d) >= 30 else {}

    score_1h = confluence_score(ind_1h) if ind_1h else 0
    score_6h = confluence_score(ind_6h) if ind_6h else 0
    score_1d = confluence_score(ind_1d) if ind_1d else 0

    signs = [score_1h > 0, score_6h > 0, score_1d > 0]
    agreement = sum(signs)

    if agreement >= 2 and score_1h > 30:
        direction = "bullish"
        confidence = min(0.5 + (score_1h / 200), 0.95)
    elif (3 - agreement) >= 2 and score_1h < -30:
        direction = "bearish"
        confidence = min(0.5 + (abs(score_1h) / 200), 0.95)
    else:
        direction = "neutral"
        confidence = 0.3

    return {
        "direction": direction,
        "confidence": round(confidence, 2),
        "scores": {"1h": score_1h, "6h": score_6h, "1d": score_1d},
        "indicators_1h": ind_1h,
        "indicators_6h": ind_6h,
        "indicators_1d": ind_1d,
    }


def mean_reversion_signal(indicators):
    """Generate mean reversion signal for range-bound markets (ADX < 25)."""
    adx = indicators.get("adx")
    rsi = indicators.get("rsi")
    bb_pct = indicators.get("bb_pct")
    stoch_k = indicators.get("stoch_rsi_k")
    price = indicators.get("price")
    bb_middle = indicators.get("bb_middle")
    atr = indicators.get("atr")
    vol_ratio = indicators.get("vol_ratio", 1.0)

    if not all(v is not None for v in [rsi, bb_pct, stoch_k, price, atr]):
        return None

    # LONG: oversold at lower BB
    if bb_pct < 0.1 and rsi < 35 and stoch_k < 0.2:
        confidence = 0.65
        if rsi < 25: confidence += 0.05
        if bb_pct < 0.0: confidence += 0.05
        if stoch_k < 0.05: confidence += 0.05
        if adx is not None and adx < 20: confidence += 0.05  # Range-bound = better for MR

        target = bb_middle if bb_middle else price * 1.025
        return {
            "action": "buy",
            "confidence": min(confidence, 0.90),
            "entry_price": price,
            "stop_loss": round(price - (atr * 1.2), 2),
            "take_profit": round(target, 2),
            "strategy": "mean_reversion",
            "reasoning": f"Mean reversion BUY: BB%={bb_pct:.3f} RSI={rsi:.0f} StochK={stoch_k:.2f} ADX={adx}",
        }

    # SHORT: overbought at upper BB
    if bb_pct > 0.9 and rsi > 65 and stoch_k > 0.8:
        confidence = 0.65
        if rsi > 75: confidence += 0.05
        if bb_pct > 1.0: confidence += 0.05
        if stoch_k > 0.95: confidence += 0.05

        target = bb_middle if bb_middle else price * 0.975
        return {
            "action": "sell",
            "confidence": min(confidence, 0.90),
            "entry_price": price,
            "stop_loss": round(price + (atr * 1.2), 2),
            "take_profit": round(target, 2),
            "strategy": "mean_reversion",
            "reasoning": f"Mean reversion SELL: BB%={bb_pct:.3f} RSI={rsi:.0f} StochK={stoch_k:.2f}",
        }

    return None


def detect_liquidation_cascade(candles_1h, current_indicators):
    """Detect liquidation cascade reversal using price action as proxy."""
    import numpy as np

    if len(candles_1h) < 24:
        return None

    closes = [c["close"] for c in candles_1h]
    volumes = [c["volume"] for c in candles_1h]

    price_now = closes[-1]
    price_4h_ago = closes[-4] if len(closes) >= 4 else closes[0]
    move_4h_pct = abs(price_now - price_4h_ago) / price_4h_ago * 100

    atr_pct = current_indicators.get("atr_pct", 2.0)

    vol_mean = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    vol_current = volumes[-1]
    vol_spike = vol_current / vol_mean if vol_mean > 0 else 1.0

    is_cascade = move_4h_pct > atr_pct * 2 and vol_spike > 2.5

    if not is_cascade:
        return None

    cascade_direction = "down" if price_now < price_4h_ago else "up"

    last_candle = candles_1h[-1]
    if cascade_direction == "down":
        reversal = last_candle["close"] > last_candle["open"]
    else:
        reversal = last_candle["close"] < last_candle["open"]

    if not reversal:
        return None

    total_vp = sum(c["close"] * c["volume"] for c in candles_1h[-24:])
    total_v = sum(c["volume"] for c in candles_1h[-24:])
    vwap_24h = total_vp / total_v if total_v > 0 else price_now

    atr = current_indicators.get("atr", price_now * 0.02)

    if cascade_direction == "down":
        return {
            "action": "buy",
            "confidence": 0.75,
            "size_pct": 10,
            "entry_price": price_now,
            "stop_loss": round(min(c["low"] for c in candles_1h[-4:]) - atr * 0.5, 2),
            "take_profit": round(vwap_24h, 2),
            "strategy": "liquidation_reversal",
            "reasoning": f"Cascade reversal: {move_4h_pct:.1f}% drop, vol {vol_spike:.1f}x, "
                        f"reversal candle. Target VWAP ${vwap_24h:,.0f}",
        }
    else:
        return {
            "action": "sell",
            "confidence": 0.70,
            "strategy": "liquidation_reversal",
            "reasoning": f"Squeeze exhaustion: {move_4h_pct:.1f}% pump, vol {vol_spike:.1f}x",
        }


def quick_signal(indicators):
    """Generate a quick technical signal summary (bullish/bearish/neutral)."""
    if not indicators:
        return "neutral", 0.5

    score = 0
    factors = 0

    rsi = indicators.get("rsi")
    if rsi is not None:
        factors += 1
        if rsi < 30:
            score += 1  # Oversold = bullish
        elif rsi > 70:
            score -= 1  # Overbought = bearish

    macd_cross = indicators.get("macd_crossover")
    if macd_cross == "bullish":
        score += 1.5
        factors += 1
    elif macd_cross == "bearish":
        score -= 1.5
        factors += 1
    else:
        factors += 1

    ema_trend = indicators.get("ema_trend")
    if ema_trend == "bullish":
        score += 1
        factors += 1
    elif ema_trend == "bearish":
        score -= 1
        factors += 1
    else:
        factors += 1

    bb_pct = indicators.get("bb_pct")
    if bb_pct is not None:
        factors += 1
        if bb_pct < 0.1:
            score += 0.5  # Near lower band
        elif bb_pct > 0.9:
            score -= 0.5  # Near upper band

    vol_ratio = indicators.get("vol_ratio")
    if vol_ratio is not None and vol_ratio > 1.5:
        score *= 1.2  # High volume amplifies signal

    if factors == 0:
        return "neutral", 0.5

    normalized = score / (factors * 1.5)  # Normalize to -1 to 1
    if normalized > 0.2:
        return "bullish", min(0.5 + normalized * 0.5, 1.0)
    elif normalized < -0.2:
        return "bearish", min(0.5 + abs(normalized) * 0.5, 1.0)
    return "neutral", 0.5
