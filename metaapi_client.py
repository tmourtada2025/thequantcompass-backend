import os
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

metaapi = MetaApi(API_TOKEN)

async def fetch_prices():
    try:
        accounts = await metaapi.metatrader_account_api.get_all_accounts()  # ✅ Updated method
        if not accounts:
            return {"error": "No MetaTrader accounts found."}

        account = accounts[0]
        await account.load()
        connection = account.get_streaming_connection()
        await connection.connect()
        await connection.wait_synchronized()

        price = await connection.get_price('XAUUSD')
        return price
    except Exception as e:
        return {"error": str(e)}
