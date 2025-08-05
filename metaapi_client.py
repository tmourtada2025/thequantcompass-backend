import os
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi  # ✅ Fixed import

load_dotenv()

# Load MetaApi credentials from environment variables
TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")  # Optional manual override

# Symbols to fetch prices for
SYMBOLS = ['US30.cash', 'XAUUSD']


async def fetch_prices():
    if not TOKEN:
        raise ValueError("METAAPI_TOKEN not found in environment variables.")

    metaapi = MetaApi(TOKEN)
    print("Connecting to MetaApi...")

    try:
        accounts = await metaapi.metatrader_account_api.get_accounts()
    except Exception as e:
        raise Exception(f"Failed to retrieve accounts: {e}")

    # Prefer manually set account, or fallback to first deployed
    account = None
    if ACCOUNT_ID:
        account = next((acc for acc in accounts if acc['id'] == ACCOUNT_ID and acc['state'] == 'DEPLOYED'), None)
    else:
        account = next((acc for acc in accounts if acc['state'] == 'DEPLOYED'), None)

    if not account:
        raise Exception("No deployed MetaTrader account found.")

    account_id = account['id']
    print(f"Connected to MetaTrader account: {account_id}")

    try:
        connection = await metaapi.metatrader_account_api.get_account(account_id)
        connection = await connection.connect()
        await connection.wait_synchronized()
    except Exception as e:
        raise Exception(f"Error synchronizing with MetaTrader account: {e}")

    prices = {}

    for symbol in SYMBOLS:
        try:
            price = await connection.get_symbol_price(symbol)
            prices[symbol] = {
                'bid': price['bid'],
                'ask': price['ask'],
                'time': price['time']
            }
        except Exception as e:
            prices[symbol] = {'error': f"Failed to fetch {symbol}: {str(e)}"}

    return prices
