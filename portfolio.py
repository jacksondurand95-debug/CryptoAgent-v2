"""Portfolio tracking — positions, P&L, trailing stops, trade log.

v2: No file I/O. Accept state dict, return modified state dict.
The caller handles persistence (GitHub Gist).
"""
import logging
import time

log = logging.getLogger("portfolio")

# v2: Ultra aggressive — minimal restrictions
MIN_HOLD_MINUTES = 5
REENTRY_COOLDOWN_MINUTES = 5
ROUND_TRIP_FEE_PCT = 0.008  # Limit order fees

# Trailing stop config
TRAILING_ACTIVATION_R = 0.5   # Activate at 0.5R profit
TRAILING_STOP_RATIO = 0.67   # Trail at 67% of initial risk distance from peak
TIME_STOP_HOURS = 24          # Cut losers after 24h if < 0.5R


def new_state():
    """Create a fresh empty state dict."""
    return {
        "positions": [],
        "trades": [],
        "starting_value": None,
        "pending_orders": [],
    }


def open_position(state, pair, side, entry_price, qty, usd_amount, stop_loss, take_profit):
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
        "highest_price": entry_price,
        "lowest_price": entry_price,
        "trailing_active": False,
        "opened_at": time.time(),
        "opened_at_str": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    state["positions"].append(pos)
    log.info(
        f"OPEN: {side} {qty:.6f} {pair} @ ${entry_price:,.2f} "
        f"SL=${stop_loss:,.2f} TP=${take_profit:,.2f}"
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

    # Keep last 500 trades
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
    """Check all positions for stop loss, take profit, trailing stops, and time stops.

    Args:
        state: The portfolio state dict
        get_price_fn: Callable(pair) -> float or None

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
        initial_risk = abs(entry - pos["initial_stop"])

        if initial_risk <= 0:
            initial_risk = entry * 0.02  # Fallback: 2% risk

        # Update highest/lowest price seen
        if price > pos.get("highest_price", entry):
            pos["highest_price"] = price
        if price < pos.get("lowest_price", entry):
            pos["lowest_price"] = price

        # ── TRAILING STOP LOGIC ──
        if side == "long":
            # Activate trailing stop after price moves enough in our favor
            if not pos.get("trailing_active"):
                profit_r = (price - entry) / initial_risk
                if profit_r >= TRAILING_ACTIVATION_R:
                    pos["trailing_active"] = True
                    log.info(f"[TRAIL] {pos['pair']}: Trailing stop activated at {profit_r:.1f}R")

            # Update trailing stop (long)
            if pos.get("trailing_active"):
                highest = pos.get("highest_price", price)
                trail_distance = initial_risk * TRAILING_STOP_RATIO
                new_stop = round(highest - trail_distance, 2)
                if new_stop > pos["stop_loss"]:
                    old_stop = pos["stop_loss"]
                    pos["stop_loss"] = new_stop
                    log.info(
                        f"[TRAIL] {pos['pair']}: Stop raised ${old_stop:,.2f} -> ${new_stop:,.2f} "
                        f"(highest=${highest:,.2f})"
                    )

            # ── STOP LOSS (long) ──
            if price <= pos["stop_loss"]:
                reason = "trailing_stop" if pos.get("trailing_active") else "stop_loss"
                state, trade = close_position(state, pos["pair"], price, reason=reason)
                if trade:
                    closed.append(trade)
                continue

            # ── TAKE PROFIT (long) ──
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
                    log.info(f"[TRAIL] {pos['pair']}: Short trailing stop activated at {profit_r:.1f}R")

            if pos.get("trailing_active"):
                lowest = pos.get("lowest_price", price)
                trail_distance = initial_risk * TRAILING_STOP_RATIO
                new_stop = round(lowest + trail_distance, 2)
                if new_stop < pos["stop_loss"]:
                    old_stop = pos["stop_loss"]
                    pos["stop_loss"] = new_stop
                    log.info(
                        f"[TRAIL] {pos['pair']}: Short stop lowered ${old_stop:,.2f} -> ${new_stop:,.2f} "
                        f"(lowest=${lowest:,.2f})"
                    )

            # Stop loss (short)
            if price >= pos["stop_loss"]:
                reason = "trailing_stop" if pos.get("trailing_active") else "stop_loss"
                state, trade = close_position(state, pos["pair"], price, reason=reason)
                if trade:
                    closed.append(trade)
                continue

            # Take profit (short)
            if price <= pos["take_profit"]:
                state, trade = close_position(state, pos["pair"], price, reason="take_profit")
                if trade:
                    closed.append(trade)
                continue

        # ── TIME STOP (both sides) ──
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

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl_usd": round(total_pnl, 2),
        "total_fees_usd": round(total_fees, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "best_trade": round(max(t["pnl_usd"] for t in trades), 2),
        "worst_trade": round(min(t["pnl_usd"] for t in trades), 2),
        "profit_factor": round(
            abs(sum(t["pnl_usd"] for t in wins) / loss_total), 2
        ) if losses and loss_total != 0 else float("inf"),
    }
