from fastapi import Query
import httpx, time, os

METAAPI_TOKEN  = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID     = os.getenv("ACCOUNT_ID") or os.getenv("METAAPI_ACCOUNT_ID")

BASE_URL = "https://mt-client-api-v1.agiliumtrade.ai"

async def _call_metaapi(path: str, params: dict | None = None):
    """Helper for authenticated GETs against MetaApi REST."""
    headers = {
        "Content-Type": "application/json",
        "auth-token": METAAPI_TOKEN
    }
    async with httpx.AsyncClient(timeout=15) as client:
        url = f"{BASE_URL}{path}"
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()

@app.get("/candles")
async def candles(
    symbol:    str  = Query(..., examples=["US30.cash", "XAUUSD"]),
    timeframe: str  = Query("1m", description="1m, 5m, 1h, 1d, …"),
    limit:     int  = Query(100, ge=1, le=1000,
                            description="Number of bars to return")
):
    """
    Return recent candle (OHLCV) data from MetaApi.\n
      • **symbol** – MT4/MT5 symbol name  \n
      • **timeframe** – MetaApi timeframe code (1m, 1h, 1d …)  \n
      • **limit** – how many bars (max 1000)\n
    """
    if not (METAAPI_TOKEN and ACCOUNT_ID):
        return {"error": "Server missing METAAPI_TOKEN or ACCOUNT_ID env vars"}

    path = (
        f"/users/current/accounts/{ACCOUNT_ID}"
        f"/symbols/{symbol}/timeframes/{timeframe}/candles"
    )

    # MetaApi uses `limit` & `timestamp` (ms since epoch) as optional query params
    params = {"limit": limit, "timestamp": int(time.time() * 1000)}
    try:
        data = await _call_metaapi(path, params)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": data  # list of dicts with time, open, high, low, close, volume
        }
    except httpx.HTTPStatusError as e:
        return {"error": f"{e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}
