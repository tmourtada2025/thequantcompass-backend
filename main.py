# main.py
from fastapi import FastAPI
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/")
def home():
    return {"status": "TradeEdge backend is live"}

@app.get("/prices")
def prices():          # regular (non-async) function
    return fetch_prices()
