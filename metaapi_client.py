import os
import asyncio
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi

# Load environment variables
load_dotenv()

# Read sensitive config from environment
META_API_TOKEN = os.getenv("META_API_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

# Fail early if any are missing
if not META_API_TOKEN or not ACCOUNT_ID:
    raise ValueError("❌ Missing META_API_TOKEN or ACCOUNT_ID in environment variables.")

# Initialize MetaApi
metaapi = MetaApi(META_API_TOKEN)

async def fetch_prices():
    try:
        account = await metaapi.metatrader_account_api.get_account(ACCOUNT_ID)
        if account.state != 'DEPLOYED':
            await account.deploy()
            await account.wait_deployed()

        connection = account.get_streaming_connection()
        await connection.connect()
        await connection.wait_synchronized()

        await connection.subscribe_to_market_data('US30')
        price = connection.price('US30')

        return {
            "symbol": "US30",
            "bid": price['bid'],
            "ask": price['ask']
        }

    except Exception as e:
        return {"error": str(e)}

# For local testing
if __name__ == "__main__":
    result = asyncio.run(fetch_prices())
    print(result)
