from fastapi import FastAPI
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/")
def root():
    return {"status": "TradeEdge backend is live"}

@app.get("/prices")
async def prices():
    return await fetch_prices()     # fetch_prices is async now
