import os
from dotenv import load_dotenv
from metaapi.cloud_metaapi_sdk import MetaApi  # ✅ Corrected import

# Load environment variables
load_dotenv()

META_API_TOKEN = os.getenv("META_API_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

if not META_API_TOKEN or not ACCOUNT_ID:
    raise ValueError("❌ Missing META_API_TOKEN or ACCOUNT_ID in environment variables.")

metaapi = MetaApi(META_API_TOKEN)

async def fetch_prices():
    account = await metaapi.metatrader_account_api.get_account(ACCOUNT_ID)
    if account.state != 'DEPLOYED':
        print(f'Account {ACCOUNT_ID} is not deployed yet. Deploying now...')
        await account.deploy()
        await account.wait_until_deployed()

    connection = account.get_streaming_connection()
    await connection.connect()
    await connection.wait_synchronized()

    us30_price = await connection.subscribe_to_market_data('US30')
    return {"symbol": "US30", "price": us30_price}
