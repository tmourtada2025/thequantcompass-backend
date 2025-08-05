# metaapi_client.py  (async version)
import os, httpx, asyncio
from dotenv import load_dotenv

load_dotenv()
TOKEN       = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID  = os.getenv("METAAPI_ACCOUNT_ID")
BASE_URL    = "https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai"

async def fetch_prices(symbol: str = "US30.cash"):
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/price"
    headers = {"Authorization": f"Bearer {TOKEN}"}

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()

    return {
        "symbol": symbol,
        "bid":   data.get("bid"),
        "ask":   data.get("ask"),
        "time":  data.get("time")
    }
