from fastapi import FastAPI
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/prices")
async def prices():
    return await fetch_prices()
