import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")

def fetch_prices():
    try:
        symbol = "US30.cash"
        url = f"https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/price"

        headers = {
            "Authorization": f"Bearer {TOKEN}"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        return {
            "symbol": symbol,
            "bid": data.get("bid"),
            "ask": data.get("ask"),
            "time": data.get("time")
        }

    except Exception as e:
        return {"error": str(e)}
