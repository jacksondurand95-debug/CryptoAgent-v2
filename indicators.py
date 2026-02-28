"""Technical analysis indicators — v2 with Keltner Squeeze + Momentum.

Includes:
- Standard indicators (RSI, MACD, EMA, BB, ATR, ADX, StochRSI, VWAP)
- Keltner Channel + Squeeze detection (BB inside KC)
- Time-series momentum (AdaptiveTrend paper)
- Multi-timeframe confluence scoring
"""
import numpy as np
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
    df["bb_pct"] = bb.bollinger_pband()

    # ATR (14-period)
    atr_ind = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr_ind.average_true_range()

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

    # ── KELTNER CHANNEL (EMA20 + 1.5x ATR) ──
    kc_ema = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    kc_atr = ta.volatility.AverageTrueRange(high, low, close, window=20).average_true_range()
    df["kc_upper"] = kc_ema + 1.5 * kc_atr
    df["kc_lower"] = kc_ema - 1.5 * kc_atr
    df["kc_middle"] = kc_ema

    # ── SQUEEZE DETECTION: BB inside KC ──
    df["squeeze"] = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])

    # Count consecutive squeeze bars
    squeeze_count = 0
    squeeze_counts = []
    for val in df["squeeze"]:
        if val:
            squeeze_count += 1
        else:
            squeeze_count = 0
        squeeze_counts.append(squeeze_count)
    df["squeeze_bars"] = squeeze_counts

    # ── TIME-SERIES MOMENTUM (AdaptiveTrend) ──
    # Momentum over lookback period
    for lb in [7, 14, 28]:
        if len(df) > lb:
            df[f"momentum_{lb}"] = (close - close.shift(lb)) / close.shift(lb)
        else:
            df[f"momentum_{lb}"] = 0.0

    # Rolling Sharpe ratio (proxy for trend quality)
    returns = close.pct_change()
    if len(df) >= 28:
        df["rolling_sharpe_28"] = (
            returns.rolling(28).mean() / returns.rolling(28).std()
        ) * np.sqrt(365 * 4)  # Annualized for 6H candles
    else:
        df["rolling_sharpe_28"] = 0.0

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
        "bb_middle": round(latest["bb_middle"], 2) if pd.notna(latest["bb_middle"]) else None,
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
        # Keltner Channel
        "kc_upper": round(latest["kc_upper"], 2) if pd.notna(latest["kc_upper"]) else None,
        "kc_lower": round(latest["kc_lower"], 2) if pd.notna(latest["kc_lower"]) else None,
        "kc_middle": round(latest["kc_middle"], 2) if pd.notna(latest["kc_middle"]) else None,
        # Squeeze
        "squeeze": bool(latest["squeeze"]) if pd.notna(latest["squeeze"]) else False,
        "squeeze_bars": int(latest["squeeze_bars"]),
        "squeeze_releasing": bool(prev["squeeze"] and not latest["squeeze"]),
        # Momentum (AdaptiveTrend)
        "momentum_7": round(float(latest["momentum_7"]), 4) if pd.notna(latest["momentum_7"]) else 0,
        "momentum_14": round(float(latest["momentum_14"]), 4) if pd.notna(latest["momentum_14"]) else 0,
        "momentum_28": round(float(latest["momentum_28"]), 4) if pd.notna(latest["momentum_28"]) else 0,
        "rolling_sharpe": round(float(latest["rolling_sharpe_28"]), 2) if pd.notna(latest["rolling_sharpe_28"]) else 0,
        # Price action
        "24h_change_pct": round((latest["close"] - df.iloc[-24]["close"]) / df.iloc[-24]["close"] * 100, 2) if len(df) >= 24 else None,
        "high_24h": round(df["high"].tail(24).max(), 2) if len(df) >= 24 else None,
        "low_24h": round(df["low"].tail(24).min(), 2) if len(df) >= 24 else None,
    }

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

    # Momentum boost
    mom = indicators.get("momentum_28", 0)
    if mom > 0.05:
        score += 15
    elif mom < -0.05:
        score -= 15

    # Squeeze release boost
    if indicators.get("squeeze_releasing"):
        score = int(score * 1.3)

    return max(-100, min(100, score))


def multi_tf_confluence(candles_1h, candles_6h, candles_1d):
    """Score across 3 timeframes, return direction + confidence + indicators."""
    ind_1h = compute_all(candles_1h) if len(candles_1h) >= 30 else {}
    ind_6h = compute_all(candles_6h) if len(candles_6h) >= 30 else {}
    ind_1d = compute_all(candles_1d) if len(candles_1d) >= 30 else {}

    score_1h = confluence_score(ind_1h) if ind_1h else 0
    score_6h = confluence_score(ind_6h) if ind_6h else 0
    score_1d = confluence_score(ind_1d) if ind_1d else 0

    # 6H is primary, 1D is trend filter, 1H is timing
    weighted = score_6h * 0.5 + score_1d * 0.3 + score_1h * 0.2

    if weighted > 25:
        direction = "bullish"
        confidence = min(0.5 + (weighted / 150), 0.95)
    elif weighted < -25:
        direction = "bearish"
        confidence = min(0.5 + (abs(weighted) / 150), 0.95)
    else:
        direction = "neutral"
        confidence = 0.3

    return {
        "direction": direction,
        "confidence": round(confidence, 2),
        "scores": {"1h": score_1h, "6h": score_6h, "1d": score_1d},
        "weighted_score": round(weighted, 1),
        "indicators_1h": ind_1h,
        "indicators_6h": ind_6h,
        "indicators_1d": ind_1d,
    }
