import os
from dotenv import load_dotenv
from metaapi.cloud_metaapi_sdk import MetaApi

load_dotenv()

TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")
SYMBOLS = ['US30.cash', 'XAUUSD']

async def fetch_prices():
    metaapi = MetaApi(TOKEN)

    try:
        print(f"Connecting to account: {ACCOUNT_ID}")
        account = await metaapi.metatrader_account_api.get_account(ACCOUNT_ID)
        connection = await account.get_rpc_connection()
        await connection.connect()
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

    except Exception as e:
        return {"error": f"Failed to connect: {str(e)}"}
