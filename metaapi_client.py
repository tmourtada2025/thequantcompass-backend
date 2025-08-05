import os
from metaapi.cloud_metaapi_sdk import MetaApi

# Read from Render environment variables directly
TOKEN = os.getenv("META_API_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

if not TOKEN or not ACCOUNT_ID:
    raise ValueError("❌ Missing META_API_TOKEN or ACCOUNT_ID in environment variables.")

SYMBOLS = ['US30.cash', 'XAUUSD']

async def fetch_prices():
    metaapi = MetaApi(TOKEN)

    print("Connecting to MetaAPI...")
    accounts = await metaapi.metatrader_account_api.get_accounts()

    # Use the first connected and deployed account
    account = next((acc for acc in accounts if acc['state'] == 'DEPLOYED'), None)

    if not account:
        raise Exception("No deployed MetaTrader accounts found.")

    account_id = account['id']
    print(f"Connected to account ID: {account_id}")

    connection = await metaapi.metatrader_account_api.get_account(account_id)
    connection = await connection.connect()
    await connection.wait_synchronized()

    results = {}

    for symbol in SYMBOLS:
        try:
            price = await connection.get_symbol_price(symbol)
            results[symbol] = {
                'bid': price['bid'],
                'ask': price['ask'],
                'time': price['time']
            }
        except Exception as e:
            results[symbol] = {'error': str(e)}

    return results
