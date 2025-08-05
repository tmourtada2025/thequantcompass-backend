import os
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

load_dotenv()

API_TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")

if not API_TOKEN or not ACCOUNT_ID:
    raise ValueError("Missing METAAPI_TOKEN or METAAPI_ACCOUNT_ID environment variables")

api = MetaApi(API_TOKEN)

async def fetch_prices():
    try:
        # retrieve account by ID
        account = await api.metatrader_account_api.get_account(account_id=ACCOUNT_ID)
        
        # wait until it's deployed and synchronized
        if account.state != 'DEPLOYED':
            await account.deploy()
        await account.wait_connected()

        connection = account.get_streaming_connection()

        await connection.connect()
        await connection.wait_synchronized()

        # Official API: get symbol price via websocket streaming
        price = await connection.get_symbol_price("US30.cash")
        return {
            "symbol": price.get("symbol"),
            "bid": price.get("bid"),
            "ask": price.get("ask"),
            "time": price.get("time")
        }

    except Exception as e:
        return {"error": str(e)}
