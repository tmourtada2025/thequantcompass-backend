from fastapi import FastAPI
import asyncio
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/prices")
async def get_prices():
    try:
        prices = await fetch_prices()
        return prices
    except Exception as e:
        return {"error": str(e)}
