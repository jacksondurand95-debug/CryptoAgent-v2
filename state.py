"""State management via GitHub Gist — serverless persistence.

Replaces fly.io persistent volume with a private GitHub Gist.
State includes open positions, trade history, and portfolio tracking.
"""
import json
import logging
import time

import requests

import config

log = logging.getLogger("state")


class GistState:
    """Read/write agent state to a private GitHub Gist."""

    EMPTY_STATE = {
        "positions": [],
        "trades": [],
        "starting_value": None,
        "peak_value": None,
        "pending_orders": [],
        "last_updated": None,
    }

    def __init__(self):
        self.gist_id = config.GIST_ID
        self.token = config.GITHUB_TOKEN
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self._cache = None

    def load(self):
        """Load state from Gist with 3-retry."""
        if not self.gist_id or not self.token:
            log.warning("No Gist configured, using empty state")
            return dict(self.EMPTY_STATE)

        for attempt in range(3):
            try:
                r = requests.get(
                    f"https://api.github.com/gists/{self.gist_id}",
                    headers=self.headers,
                    timeout=10,
                )
                r.raise_for_status()
                files = r.json().get("files", {})
                if "state.json" not in files:
                    log.warning("state.json not found in Gist, returning empty state")
                    return dict(self.EMPTY_STATE)

                content = files["state.json"]["content"]
                self._cache = json.loads(content)

                # Ensure all expected keys exist (forward compatibility)
                for key, default in self.EMPTY_STATE.items():
                    if key not in self._cache:
                        self._cache[key] = default

                return self._cache

            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    log.warning(f"State load failed (attempt {attempt+1}): {e} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    log.error(f"Failed to load state after 3 attempts: {e}")
                    return dict(self.EMPTY_STATE)

    def save(self, state):
        """Save state to Gist with 3-retry."""
        if not self.gist_id or not self.token:
            log.warning("No Gist configured, state not saved")
            return False

        # Keep only last 200 trades to prevent Gist from getting huge
        if "trades" in state:
            state["trades"] = state["trades"][-200:]

        # Timestamp the save
        state["last_updated"] = time.time()

        for attempt in range(3):
            try:
                r = requests.patch(
                    f"https://api.github.com/gists/{self.gist_id}",
                    headers=self.headers,
                    json={
                        "files": {
                            "state.json": {
                                "content": json.dumps(state, indent=2)
                            }
                        }
                    },
                    timeout=10,
                )
                r.raise_for_status()
                self._cache = state
                return True

            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    log.warning(f"State save failed (attempt {attempt+1}): {e} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    log.error(f"Failed to save state after 3 attempts: {e}")
                    return False

    def get_cached(self):
        """Return cached state without hitting the API."""
        if self._cache is not None:
            return self._cache
        return self.load()

    def add_position(self, state, position):
        """Add a new position to state and save."""
        state["positions"].append(position)
        return self.save(state)

    def remove_position(self, state, pair):
        """Remove a position by pair and save."""
        state["positions"] = [p for p in state["positions"] if p.get("pair") != pair]
        return self.save(state)

    def add_trade(self, state, trade):
        """Record a completed trade and save."""
        state["trades"].append(trade)
        return self.save(state)

    def add_pending_order(self, state, order_info):
        """Track a pending limit order."""
        state.setdefault("pending_orders", [])
        state["pending_orders"].append(order_info)
        return self.save(state)

    def remove_pending_order(self, state, order_id):
        """Remove a pending order by its ID."""
        state.setdefault("pending_orders", [])
        state["pending_orders"] = [
            o for o in state["pending_orders"]
            if o.get("order_id") != order_id
        ]
        return self.save(state)

    def get_position(self, state, pair):
        """Get position for a specific pair, or None."""
        for p in state.get("positions", []):
            if p.get("pair") == pair:
                return p
        return None

    def update_peak(self, state, current_value):
        """Update peak portfolio value for drawdown tracking."""
        if state.get("starting_value") is None:
            state["starting_value"] = current_value
        peak = state.get("peak_value") or 0
        if current_value > peak:
            state["peak_value"] = current_value
            self.save(state)

    @staticmethod
    def create_gist(token):
        """One-time setup: create a new private Gist for state storage.

        Run this once to get the GIST_ID, then set STATE_GIST_ID env var.
        Usage: python -c "from state import GistState; GistState.create_gist('ghp_...')"
        """
        initial = {
            "positions": [],
            "trades": [],
            "starting_value": None,
            "peak_value": None,
            "pending_orders": [],
            "last_updated": time.time(),
        }
        r = requests.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={
                "description": "CryptoAgent v2 State",
                "public": False,
                "files": {
                    "state.json": {
                        "content": json.dumps(initial, indent=2)
                    }
                },
            },
            timeout=10,
        )
        r.raise_for_status()
        gist_data = r.json()
        gist_id = gist_data["id"]
        gist_url = gist_data["html_url"]
        print(f"Created Gist: {gist_id}")
        print(f"URL: {gist_url}")
        print(f"\nSet this as your GitHub Actions secret:")
        print(f"  STATE_GIST_ID={gist_id}")
        return gist_id
