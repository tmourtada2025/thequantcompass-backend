from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "TradeEdge backend is live"}
from fastapi import APIRouter
import asyncio
from metaapi_client import fetch_prices

@app.get("/prices")
async def get_prices():
    try:
        prices = await fetch_prices()
        return {"prices": prices}
    except Exception as e:
        return {"error": str(e)}
