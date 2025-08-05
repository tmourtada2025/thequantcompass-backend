# metaapi_client.py
import os, asyncio, httpx, time
from fastapi import HTTPException

# Pull secrets that Render already has
TOKEN  = os.getenv("METAAPI_TOKEN")
ACC_ID = os.getenv("METAAPI_ACCOUNT_ID")

if not TOKEN or not ACC_ID:
    raise RuntimeError("METAAPI_TOKEN or METAAPI_ACCOUNT_ID is missing")

# Use the newer, always-resolvable hostname
BASE_URL = "https://mt-client-api-v1.metaapi.cloud"

HEADERS = {"auth-token": TOKEN}

async def _call_metaapi(path: str, params: dict | None = None) -> dict:
    """
    Helper that performs an HTTP GET with automatic retries +
    nice error messages instead of 500s.
    """
    url     = f"{BASE_URL}{path}"
    retries = 3
    backoff = 0.7

    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(retries):
            try:
                r = await client.get(url, headers=HEADERS, params=params)
                r.raise_for_status()
                return r.json()

            except httpx.RequestError as e:
                if attempt == retries - 1:
                    raise HTTPException(
                        502,
                        detail=f"DNS / network error contacting MetaApi: {e}"
                    )
                time.sleep(backoff)
                backoff *= 2

            except httpx.HTTPStatusError as e:
                # MetaApi gives good JSON bodies – surface them
                raise HTTPException(
                    e.response.status_code,
                    detail=e.response.json()
                ) from None


async def fetch_prices(symbol: str = "US30.cash") -> dict:
    """
    Returns the live bid / ask for a single symbol.
    """
    data = await _call_metaapi(
        f"/users/current/accounts/{ACC_ID}/symbols/{symbol}/price"
    )
    # MetaApi returns an array under 'prices'
    if not data or not data.get("price"):
        raise HTTPException(503, detail="Price feed temporarily unavailable")
    return data["price"]    # {'bid': .., 'ask': .., 'time': ..}


async def fetch_candles(
    symbol: str,
    timeframe: str = "1m",
    limit: int = 100
) -> list[dict]:
    """
    Returns the most-recent `limit` candles (MetaApi calls them 'bars').
    """
    params = {"timeframe": timeframe, "limit": limit}
    data   = await _call_metaapi(
        f"/users/current/accounts/{ACC_ID}/symbols/{symbol}/candles",
        params=params
    )
    return data.get("candles", [])
