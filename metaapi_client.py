from metaapi_cloud_metaapi_sdk import MetaApi
import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
metaapi = MetaApi(API_TOKEN)

async def fetch_prices():
    try:
        # ✅ Get all accounts
        accounts = await metaapi.metatrader_account_api.get_all_accounts()

        if not accounts:
            return {"error": "No MetaTrader accounts found"}

        account = accounts[0]  # Or match by ID if needed
        await account.load()

        if account.connection_status != 'CONNECTED':
            return {"error": "Account not connected"}

        connection = account.get_streaming_connection()
        await connection.connect()
        await connection.wait_synchronized(timeout_in_seconds=30)

        price = await connection.get_price('XAUUSD')

        if price:
            return {
                "symbol": "XAUUSD",
                "bid": price['bid'],
                "ask": price['ask']
            }
        else:
            return {"error": "No price data for XAUUSD"}

    except Exception as e:
        return {"error": str(e)}
