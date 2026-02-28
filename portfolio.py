"""Portfolio tracking — positions, P&L, dynamic trailing stops, trade log.

v2.1: AdaptiveTrend-style dynamic trailing stops (2.5x ATR).
Wider stops, longer holds, fewer but bigger wins.
"""
import logging
import time

import config

log = logging.getLogger("portfolio")

# From config (single source of truth)
MIN_HOLD_MINUTES = config.MIN_HOLD_MINUTES
REENTRY_COOLDOWN_MINUTES = config.REENTRY_COOLDOWN_MINUTES
ROUND_TRIP_FEE_PCT = config.ROUND_TRIP_FEE_PCT

# Trailing stop config — from AdaptiveTrend paper (2.5x ATR optimal)
TRAILING_ACTIVATION_R = config.TRAILING_ACTIVATION_R
TRAILING_ATR_MULT = config.TRAILING_STOP_ATR_MULT
TIME_STOP_HOURS = config.TIME_STOP_HOURS


def new_state():
    """Create a fresh empty state dict."""
    return {
        "positions": [],
        "trades": [],
        "starting_value": None,
        "pending_orders": [],
        "peak_value": None,
    }


def open_position(state, pair, side, entry_price, qty, usd_amount, stop_loss, take_profit, atr=None):
    """Record a new position. Returns updated state."""
    pos = {
        "pair": pair,
        "side": side,
        "entry_price": entry_price,
        "qty": qty,
        "usd_amount": usd_amount,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "initial_stop": stop_loss,
        "initial_risk": abs(entry_price - stop_loss),
        "atr": atr or abs(entry_price - stop_loss) / TRAILING_ATR_MULT,
        "highest_price": entry_price,
        "lowest_price": entry_price,
        "trailing_active": False,
        "opened_at": time.time(),
        "opened_at_str": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    state["positions"].append(pos)
    log.info(
        f"OPEN: {side} {qty:.6f} {pair} @ ${entry_price:,.2f} "
        f"SL=${stop_loss:,.2f} TP=${take_profit:,.2f} ATR=${pos['atr']:,.2f}"
    )
    return state


def close_position(state, pair, exit_price, reason="signal"):
    """Close a position and record the trade. Returns (state, trade_record)."""
    pos = None
    for p in state["positions"]:
        if p["pair"] == pair:
            pos = p
            break

    if not pos:
        log.warning(f"No open position for {pair}")
        return state, None

    state["positions"].remove(pos)

    entry = pos["entry_price"]
    side = pos.get("side", "long")

    if side == "long":
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100

    pnl_usd = pos["usd_amount"] * (pnl_pct / 100)
    fee_usd = pos["usd_amount"] * ROUND_TRIP_FEE_PCT
    pnl_usd -= fee_usd

    trade = {
        "pair": pair,
        "side": side,
        "entry_price": entry,
        "exit_price": exit_price,
        "qty": pos["qty"],
        "usd_amount": pos["usd_amount"],
        "pnl_pct": round(pnl_pct, 2),
        "pnl_usd": round(pnl_usd, 2),
        "fees_usd": round(fee_usd, 2),
        "reason": reason,
        "trailing_was_active": pos.get("trailing_active", False),
        "highest_price_seen": pos.get("highest_price", entry),
        "opened_at": pos["opened_at"],
        "closed_at": time.time(),
        "closed_at_str": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "duration_min": round((time.time() - pos["opened_at"]) / 60, 1),
    }

    state["trades"].append(trade)
    if len(state["trades"]) > 500:
        state["trades"] = state["trades"][-500:]

    sign = "+" if pnl_usd >= 0 else ""
    log.info(
        f"CLOSE: {pair} @ ${exit_price:,.2f} | "
        f"P&L: {sign}${pnl_usd:,.2f} ({sign}{pnl_pct:.1f}%) | "
        f"Reason: {reason} | Held: {trade['duration_min']:.0f}min"
    )
    return state, trade


def check_stops(state, get_price_fn):
    """Check all positions for stop loss, take profit, dynamic trailing stops, and time stops.

    Uses AdaptiveTrend-style dynamic trailing:
        trailing_stop = max(prev_stop, current_price - 2.5 * ATR)

    Returns:
        (updated_state, list_of_closed_trades)
    """
    closed = []
    now = time.time()

    for pos in list(state["positions"]):
        price = get_price_fn(pos["pair"])
        if price is None:
            continue

        entry = pos["entry_price"]
        side = pos.get("side", "long")
        atr = pos.get("atr", entry * 0.025)
        initial_risk = pos.get("initial_risk", abs(entry - pos["initial_stop"]))

        if initial_risk <= 0:
            initial_risk = entry * 0.02

        # Check minimum hold time
        hold_minutes = (now - pos["opened_at"]) / 60
        if hold_minutes < MIN_HOLD_MINUTES:
            continue

        # Update highest/lowest price seen
        if price > pos.get("highest_price", entry):
            pos["highest_price"] = price
        if price < pos.get("lowest_price", entry):
            pos["lowest_price"] = price

        # ── DYNAMIC TRAILING STOP (AdaptiveTrend style) ──
        if side == "long":
            # Activate trailing after reaching activation threshold
            if not pos.get("trailing_active"):
                profit_r = (price - entry) / initial_risk
                if profit_r >= TRAILING_ACTIVATION_R:
                    pos["trailing_active"] = True
                    log.info(f"[TRAIL] {pos['pair']}: Trailing activated at {profit_r:.1f}R")

            # Dynamic trailing: stop = max(prev_stop, price - 2.5 * ATR)
            if pos.get("trailing_active"):
                new_stop = round(price - TRAILING_ATR_MULT * atr, 2)
                if new_stop > pos["stop_loss"]:
                    old_stop = pos["stop_loss"]
                    pos["stop_loss"] = new_stop
                    log.info(
                        f"[TRAIL] {pos['pair']}: Stop ${old_stop:,.2f} -> ${new_stop:,.2f} "
                        f"(price=${price:,.2f} ATR=${atr:,.2f})"
                    )

            # STOP LOSS
            if price <= pos["stop_loss"]:
                reason = "trailing_stop" if pos.get("trailing_active") else "stop_loss"
                state, trade = close_position(state, pos["pair"], price, reason=reason)
                if trade:
                    closed.append(trade)
                continue

            # TAKE PROFIT
            if price >= pos["take_profit"]:
                state, trade = close_position(state, pos["pair"], price, reason="take_profit")
                if trade:
                    closed.append(trade)
                continue

        else:
            # SHORT SIDE
            if not pos.get("trailing_active"):
                profit_r = (entry - price) / initial_risk
                if profit_r >= TRAILING_ACTIVATION_R:
                    pos["trailing_active"] = True
                    log.info(f"[TRAIL] {pos['pair']}: Short trailing activated at {profit_r:.1f}R")

            if pos.get("trailing_active"):
                new_stop = round(price + TRAILING_ATR_MULT * atr, 2)
                if new_stop < pos["stop_loss"]:
                    old_stop = pos["stop_loss"]
                    pos["stop_loss"] = new_stop
                    log.info(
                        f"[TRAIL] {pos['pair']}: Short stop ${old_stop:,.2f} -> ${new_stop:,.2f}"
                    )

            if price >= pos["stop_loss"]:
                reason = "trailing_stop" if pos.get("trailing_active") else "stop_loss"
                state, trade = close_position(state, pos["pair"], price, reason=reason)
                if trade:
                    closed.append(trade)
                continue

            if price <= pos["take_profit"]:
                state, trade = close_position(state, pos["pair"], price, reason="take_profit")
                if trade:
                    closed.append(trade)
                continue

        # ── TIME STOP ──
        hours_held = (now - pos["opened_at"]) / 3600
        if hours_held > TIME_STOP_HOURS:
            if side == "long":
                current_r = (price - entry) / initial_risk
            else:
                current_r = (entry - price) / initial_risk

            if current_r < 0.5:
                state, trade = close_position(state, pos["pair"], price, reason="time_stop")
                if trade:
                    closed.append(trade)
                continue

    return state, closed


def has_position(state, pair):
    """Check if we have an open position for a pair."""
    return any(p["pair"] == pair for p in state["positions"])


def can_reenter(state, pair):
    """Check if enough time has passed since closing this pair."""
    now = time.time()
    cooldown_sec = REENTRY_COOLDOWN_MINUTES * 60
    for trade in reversed(state["trades"]):
        if trade["pair"] == pair:
            if now - trade["closed_at"] < cooldown_sec:
                return False
            break
    return True


def get_stats(trades):
    """Calculate performance stats from trade list."""
    if not trades:
        return {"total_trades": 0, "message": "No trades yet"}

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    total_fees = sum(t["fees_usd"] for t in trades)

    avg_win = sum(t["pnl_usd"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0

    loss_total = sum(t["pnl_usd"] for t in losses)

    # Average hold time
    avg_hold = sum(t["duration_min"] for t in trades) / len(trades) if trades else 0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl_usd": round(total_pnl, 2),
        "total_fees_usd": round(total_fees, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "avg_hold_min": round(avg_hold, 0),
        "best_trade": round(max(t["pnl_usd"] for t in trades), 2),
        "worst_trade": round(min(t["pnl_usd"] for t in trades), 2),
        "profit_factor": round(
            abs(sum(t["pnl_usd"] for t in wins) / loss_total), 2
        ) if losses and loss_total != 0 else float("inf"),
    }
