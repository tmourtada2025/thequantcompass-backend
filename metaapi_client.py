from metaapi_cloud_sdk import MetaApi
import os
import asyncio

# Your MetaApi token from environment variables (set this in Render's settings)
META_API_TOKEN = os.getenv("META_API_TOKEN")

# Your account ID (set this in environment variables too)
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

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

        # Example: Fetching price for US30 symbol
        quote = await connection.subscribe_to_market_data('US30')
        price = connection.price('US30')
        return {
            "symbol": "US30",
            "bid": price['bid'],
            "ask": price['ask']
        }

    except Exception as e:
        return {"error": str(e)}


# For local test
if __name__ == "__main__":
    result = asyncio.run(fetch_prices())
    print(result)
