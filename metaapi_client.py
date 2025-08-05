import os, httpx, asyncio
from typing import Dict

# --------------------------------------------------------------------
# 1.  ENVIRONMENT (must exist in Render → Environment)
# --------------------------------------------------------------------
TOKEN  = os.getenv("METAAPI_TOKEN")
ACC_ID = os.getenv("METAAPI_ACCOUNT_ID")
REGION = os.getenv("METAAPI_REGION", "london")        # london | new-york | frankfurt | ...

if not (TOKEN and ACC_ID):
    raise RuntimeError("METAAPI_TOKEN and METAAPI_ACCOUNT_ID must be set as env-vars")

BASE    = f"https://mt-client-api-v1.{REGION}.agiliumtrade.ai"
HEADERS = {"auth-token": TOKEN}


# --------------------------------------------------------------------
# 2.  Internal helper
# --------------------------------------------------------------------
async def _get(url: str) -> Dict:
    async with httpx.AsyncClient(http2=True, timeout=10) as c:
        r = await c.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------
# 3.  Called by /prices  – no arguments, returns current price for US30
# --------------------------------------------------------------------
async def fetch_prices() -> Dict:
    url = f"{BASE}/users/current/accounts/{ACC_ID}/symbols/US30/current-price"
    return await _get(url)


# --------------------------------------------------------------------
# 4.  Called by /candles  – symbol + timeframe (default 1H)
# --------------------------------------------------------------------
async def fetch_candle(symbol: str, tf: str = "1H") -> Dict:
    url = (
        f"{BASE}/users/current/accounts/{ACC_ID}"
        f"/symbols/{symbol}/candles/latest?timeframe={tf}&count=1"
    )
    data = await _get(url)
    return data[0] if isinstance(data, list) else data
