"""Shared utilities for all intel sub-agents."""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("intel")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Max entries to keep per file to prevent repo bloat
MAX_HISTORY_ENTRIES = 500


def retry_get(url, params=None, headers=None, timeout=15, retries=3):
    """HTTP GET with exponential backoff. Returns response or None."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                # Rate limited — back off harder
                wait = (2 ** attempt) * 5
                log.warning(f"Rate limited on {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            log.debug(f"HTTP {r.status_code} from {url} (attempt {attempt+1})")
        except requests.exceptions.Timeout:
            log.debug(f"Timeout from {url} (attempt {attempt+1})")
        except Exception as e:
            log.debug(f"Request error {url}: {e}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def load_data(filename):
    """Load existing JSON data file, return empty dict if missing."""
    path = DATA_DIR / filename
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_data(filename, data):
    """Save data to JSON file with timestamp and history trimming."""
    path = DATA_DIR / filename
    data["_meta"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_ts": int(time.time()),
    }
    # Trim history arrays to prevent repo bloat
    for key, val in data.items():
        if key.startswith("_"):
            continue
        if isinstance(val, list) and len(val) > MAX_HISTORY_ENTRIES:
            data[key] = val[-MAX_HISTORY_ENTRIES:]
    path.write_text(json.dumps(data, indent=2))
    log.info(f"Saved {filename} ({path.stat().st_size} bytes)")


def aggregate_signal(scores, weights=None):
    """Aggregate multiple bias scores into a single -100 to +100 score.

    scores: list of dicts with {"bias": "bullish"|"bearish"|"neutral", "weight": float}
    Returns: {"score": int, "bias": str, "strength": str}
    """
    total_score = 0
    total_weight = 0

    for s in scores:
        bias = s.get("bias", "neutral")
        weight = s.get("weight", 1.0)
        raw = s.get("score", 0)

        if bias == "bullish":
            total_score += abs(raw) * weight
        elif bias == "bearish":
            total_score -= abs(raw) * weight

        total_weight += weight

    if total_weight > 0:
        normalized = int(total_score / total_weight)
    else:
        normalized = 0

    normalized = max(-100, min(100, normalized))

    if normalized > 25:
        bias = "bullish"
    elif normalized < -25:
        bias = "bearish"
    else:
        bias = "neutral"

    strength = "strong" if abs(normalized) > 60 else "moderate" if abs(normalized) > 35 else "weak"

    return {"score": normalized, "bias": bias, "strength": strength}
