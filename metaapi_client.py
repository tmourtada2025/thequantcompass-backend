import os
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

load_dotenv()

TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")

if not TOKEN or not ACCOUNT_ID:
    raise ValueError("Missing METAAPI_TOKEN or METAAPI_ACCOUNT_ID")

api = MetaApi(TOKEN)

async def fetch_prices():
    try:
        account = await api.metatrader_account_api.get_account(account_id=ACCOUNT_ID)

        if account.state != 'DEPLOYED':
            await account.deploy()
        await account.wait_connected()

        connection = account.get_streaming_connection()
        await connection.connect()
        await connection.wait_synchronized()

        # ➤ Subscribe to symbol market data
        await connection.subscribe_to_market_data("US30.cash")

        # ➤ Then read live price via price method
        price = connection.price("US30.cash")
        return {
            "symbol": "US30.cash",
            "bid": price.get("bid"),
            "ask": price.get("ask"),
            "time": price.get("time")
        }
    except Exception as e:
        return {"error": str(e)}
