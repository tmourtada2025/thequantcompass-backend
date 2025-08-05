from fastapi import FastAPI, Query, HTTPException
from metaapi_client import fetch_prices, fetch_candle

app = FastAPI()


@app.get("/")
def root():
    """Health-check."""
    return {"status": "TradeEdge backend is live"}


# --------------------------------------------------------------------
#               ✅  SAME SIGNATURE YOU HAD BEFORE
# --------------------------------------------------------------------
@app.get("/prices")
async def prices():
    """Returns current bid/ask for US30 (Dow Jones)."""
    try:
        return await fetch_prices()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# --------------------------------------------------------------------
#               🆕  LATEST CANDLE ENDPOINT
# --------------------------------------------------------------------
@app.get("/candles")
async def candles(
    symbol: str = Query(..., description="e.g. EURUSD, XAUUSD, US30"),
    tf: str = Query("1H",  description="Timeframe: 1m, 5m, 15m, 1H, 1D …")
):
    """
    Returns the most-recent candle for the symbol/timeframe.
    Example:  /candles?symbol=EURUSD&tf=15m
    """
    try:
        return await fetch_candle(symbol, tf)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
