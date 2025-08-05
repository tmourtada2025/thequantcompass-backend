# main.py
from fastapi import FastAPI, Query
from metaapi_client import fetch_prices, fetch_candles

app = FastAPI()

@app.get("/")
def root():
    return {
        "routes": [
            "/prices?symbol=US30.cash",
            "/candles?symbol=US30.cash&timeframe=5m&limit=50"
        ]
    }


@app.get("/prices")
async def prices(symbol: str = Query("US30.cash")):
    """
    GET /prices?symbol=US30.cash
    """
    return await fetch_prices(symbol)


@app.get("/candles")
async def candles(
    symbol: str = Query(..., description="e.g. US30.cash"),
    timeframe: str = Query("1m", pattern="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    GET /candles?symbol=US30.cash&timeframe=5m&limit=50
    """
    return await fetch_candles(symbol, timeframe, limit)
