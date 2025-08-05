from fastapi import FastAPI
import uvicorn
import asyncio
from metaapi_client import fetch_prices

app = FastAPI()

@app.get("/prices")
async def get_prices():
    result = await fetch_prices()
    return {"prices": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
