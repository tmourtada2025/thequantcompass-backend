import os
import asyncio
from dotenv import load_dotenv
from metaapi.cloud_metaapi_sdk import MetaApi

load_dotenv()

API_TOKEN = os.getenv("META_API_TOKEN")
ACCOUNT_ID = os.getenv("META_API_ACCOUNT_ID")

metaapi = MetaApi(API_TOKEN)


async def fetch_prices():
    try:
        account = await metaapi.metatrader_account_api.get_account(ACCOUNT_ID)
        await account.deploy()
        await account.wait_connected()

        connection = account.get_streaming_connection()
        await connection.connect()
        await connection.wait_synchronized()

        # Fetch price for US30.cash (can be changed)
        price = await connection.get_symbol_price("US30.cash")
        return {
            "symbol": "US30.cash",
            "bid": price['bid'],
            "ask": price['ask'],
            "time": price['time']
        }

    except Exception as e:
        return {"error": str(e)}
