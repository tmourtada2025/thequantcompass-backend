from fastapi import FastAPI
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "MetaAPI Backend is Live"}

@app.get("/prices")
async def get_prices():
    return await fetch_prices()
