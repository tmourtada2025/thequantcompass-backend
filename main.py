from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "TradeEdge backend is live"}
