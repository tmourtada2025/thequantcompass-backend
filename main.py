# main.py
from fastapi import FastAPI
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/")
async def home():
    return {"status": "TradeEdge backend is live"}

@app.get("/prices")
async def prices():
    return await fetch_prices()          # ✅ now it *is* a coroutine
