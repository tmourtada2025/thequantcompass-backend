# main.py  ────────────────────────────────────────────────────────────
from fastapi import FastAPI, Query
import os, time, httpx

########################################################################
# 0. FastAPI instance  (← MUST appear before any @app.get decorator)
########################################################################
app = FastAPI()

########################################################################
# 1. Environment variables
########################################################################
METAAPI_TOKEN  = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID     = (
    os.getenv("ACCOUNT_ID")            # ← if you use this name locally
    or os.getenv("METAAPI_ACCOUNT_ID") # ← name used in Render UI
)

if not (METAAPI_TOKEN and ACCOUNT_ID):
    raise ValueError(
        "❌  METAAPI_TOKEN or ACCOUNT_ID is missing in the environment. "
        "Add them in Render ➜ Environment."
    )

########################################################################
# 2. Small helper to call MetaApi REST securely
########################################################################
BASE_URL = "https://mt-client-api-v1.agiliumtrade.ai"

async def _call_metaapi(path: str, params: dict | None = None):
    headers = {"auth-token": METAAPI_TOKEN, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=15) as client:
        url = f"{BASE_URL}{path}"
        r   = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()

########################################################################
# 3. /prices  — latest bid/ask
########################################################################
@app.get("/prices")
async def prices(symbol: str = Query("US30.cash", examples=["XAUUSD", "EURUSD"])):
    """
    Return latest bid/ask for *symbol* via MetaApi REST.
    """
    try:
        path   = f"/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/price"
        quote  = await _call_metaapi(path)
        return {
            "symbol": symbol,
            "bid":    quote["bid"],
            "ask":    quote["ask"],
            "time":   quote["time"]
        }
    except httpx.HTTPStatusError as e:
        return {"error": f"{e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

########################################################################
# 4. /candles  — OHLCV bars
########################################################################
@app.get("/candles")
async def candles(
    symbol:    str = Query(..., examples=["US30.cash", "XAUUSD"]),
    timeframe: str = Query("1m",  description="1m, 5m, 1h, 1d …"),
    limit:     int = Query(100,  ge=1, le=1000, description="Bars to return")
):
    """
    Return recent candles (OHLCV) from MetaApi.
    """
    path   = (
        f"/users/current/accounts/{ACCOUNT_ID}"
        f"/symbols/{symbol}/timeframes/{timeframe}/candles"
    )
    params = {"limit": limit, "timestamp": int(time.time() * 1000)}
    try:
        bars = await _call_metaapi(path, params)
        return {"symbol": symbol, "timeframe": timeframe, "bars": bars}
    except httpx.HTTPStatusError as e:
        return {"error": f"{e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}
