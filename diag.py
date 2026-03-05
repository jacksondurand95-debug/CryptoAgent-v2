#!/usr/bin/env python3
"""One-shot Coinbase account diagnostic. Dumps everything."""
import base64
import json
import os
import secrets
import time
from pathlib import Path

import jwt as pyjwt
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

KEY_FILE = Path(__file__).parent / "coinbase_key.json"

def get_auth():
    key_data = json.loads(KEY_FILE.read_text())
    key_id = key_data.get("name") or key_data.get("id", "")
    raw_pk = key_data.get("privateKey", "")
    if raw_pk and "BEGIN" not in raw_pk:
        raw_bytes = base64.b64decode(raw_pk)
        ed_key = Ed25519PrivateKey.from_private_bytes(raw_bytes[:32])
        pem = ed_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    else:
        pem = raw_pk.encode()
    return key_id, pem

def api_call(method, path, key_id, pem, params=None):
    uri = f"{method} api.coinbase.com{path}"
    jwt_data = {
        "sub": key_id, "iss": "cdp",
        "nbf": int(time.time()), "exp": int(time.time()) + 120,
        "uri": uri,
    }
    token = pyjwt.encode(jwt_data, pem, algorithm="EdDSA",
                         headers={"kid": key_id, "nonce": secrets.token_hex()})
    r = requests.request(method, f"https://api.coinbase.com{path}",
                         headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                         params=params, timeout=15)
    return r.status_code, r.json()

def main():
    key_id, pem = get_auth()
    print(f"KEY ID: {key_id[:20]}...")
    print()

    # 1. Transaction summary (fee tier)
    print("=" * 60)
    print("TRANSACTION SUMMARY / FEE TIER")
    print("=" * 60)
    status, data = api_call("GET", "/api/v3/brokerage/transaction_summary", key_id, pem)
    print(f"Status: {status}")
    print(json.dumps(data, indent=2))
    print()

    # 2. Account balances
    print("=" * 60)
    print("ACCOUNT BALANCES")
    print("=" * 60)
    status, data = api_call("GET", "/api/v3/brokerage/accounts", key_id, pem)
    print(f"Status: {status}")
    for acct in data.get("accounts", []):
        bal = float(acct.get("available_balance", {}).get("value", 0))
        hold = float(acct.get("hold", {}).get("value", 0))
        cur = acct.get("available_balance", {}).get("currency", "")
        name = acct.get("name", "")
        acct_uuid = acct.get("uuid", "")
        if bal > 0 or hold > 0:
            print(f"  {cur}: available={bal}, hold={hold}, name={name}, uuid={acct_uuid[:12]}...")
    print()

    # 3. Recent fills (to see actual fee rates)
    print("=" * 60)
    print("RECENT FILLS (last 10)")
    print("=" * 60)
    status, data = api_call("GET", "/api/v3/brokerage/orders/historical/fills", key_id, pem,
                            params={"limit": "10"})
    print(f"Status: {status}")
    for fill in data.get("fills", []):
        print(f"  {fill.get('trade_time', '')[:19]} | {fill.get('product_id', '')} | "
              f"{fill.get('side', '')} | price={fill.get('price', '')} | "
              f"size={fill.get('size', '')} | fee={fill.get('commission', '')} | "
              f"trade_type={fill.get('trade_type', '')}")
    print()

    # 4. Key permissions
    print("=" * 60)
    print("API KEY INFO")
    print("=" * 60)
    print(f"Key ID: {key_id}")
    print("(Permissions are encoded in the key - check CDP portal for details)")

if __name__ == "__main__":
    main()
