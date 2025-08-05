from fastapi import FastAPI
import asyncio
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/prices")
async def get_prices():
    result = await fetch_prices()
    return result
