import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()  # makes local runs easier; Render uses “Environment” panel

# ───────────────────────── CONFIG ─────────────────────────
TOKEN       = os.getenv("METAAPI_TOKEN")      # ⚠️ required
ACCOUNT_ID  = os.getenv("ACCOUNT_ID")         # ⚠️ required
REGION      = os.getenv("METAAPI_REGION", "london")  # london / new-york / etc.

BASE_URL = f"https://mt-client-api-v1.{REGION}.agiliumtrade.ai"
SYMBOLS  = ["US30.cash", "XAUUSD"]            # add more if you like
# ──────────────────────────────────────────────────────────

if not all([TOKEN, ACCOUNT_ID]):
    raise ValueError(
        "❌  METAAPI_TOKEN or ACCOUNT_ID is missing. "
        "Set them in Render → Environment."
    )

HEADERS = {
    "auth-token": TOKEN,
    "accept": "application/json",
}

async def fetch_symbol_price(client: httpx.AsyncClient, symbol: str):
    """One REST call to MetaApi → current-price endpoint."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/current-price"
    try:
        r = await client.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return symbol, {
            "bid":  data["bid"],
            "ask":  data["ask"],
            "time": data["time"],
        }
    except httpx.HTTPStatusError as e:        # 4xx / 5xx
        return symbol, {"error": f"{e.response.status_code} {e.response.text}"}
    except Exception as e:                    # network or JSON error
        return symbol, {"error": str(e)}

async def fetch_prices() -> dict:
    """Return a {symbol: {...}} dict for all symbols in SYMBOLS."""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_symbol_price(client, s) for s in SYMBOLS]
        results = await asyncio.gather(*tasks)
    return dict(results)

# run locally:  python metaapi_client.py
if __name__ == "__main__":
    print(asyncio.run(fetch_prices()))
