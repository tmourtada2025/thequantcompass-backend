#!/usr/bin/env python3
"""
Simple Polygon.io API Test
"""

import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def test_polygon_api():
    """Test basic Polygon.io API connection"""
    
    api_key = os.getenv('POLYGON_API_KEY')
    print(f"API Key: {api_key[:8]}..." if api_key else "No API key found")
    
    if not api_key:
        return
    
    # Test simple endpoint
    url = f"https://api.polygon.io/v2/aggs/ticker/C:EURUSD/prev?apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        try:
            print(f"Testing URL: {url[:50]}...")
            async with session.get(url) as response:
                print(f"Status: {response.status}")
                data = await response.json()
                print(f"Response: {data}")
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_polygon_api())
