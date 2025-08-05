from fastapi import FastAPI
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to TradeEdge Backend"}

@app.get("/prices")
async def get_prices():
    prices = await fetch_prices()
    return prices
