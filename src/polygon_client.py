import os
import json
import logging
import websocket
import threading
import time
from typing import Callable, Dict, List
from polygon import RESTClient
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PolygonClient:
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        self.rest_client = RESTClient(api_key=self.api_key) if self.api_key else None
        self.ws = None
        self.is_streaming = False
        self.callback = None
        
        # Symbols to track
        self.symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
            'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP'
        ]
        
    def is_connected(self) -> bool:
        """Check if Polygon client is properly configured"""
        return self.api_key is not None and self.rest_client is not None
    
    def get_historical_data(self, symbol: str, timespan: str = '1', 
                          multiplier: int = 1, from_date: str = None, 
                          to_date: str = None) -> List[Dict]:
        """Get historical market data"""
        try:
            if not self.rest_client:
                logger.error("Polygon client not configured")
                return []
            
            # Default to last 30 days if no dates provided
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get aggregates (OHLCV data)
            aggs = self.rest_client.get_aggs(
                ticker=f"C:{symbol}",
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date
            )
            
            bars = []
            for agg in aggs:
                bars.append({
                    'symbol': symbol,
                    'timestamp': agg.timestamp,
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': getattr(agg, 'vwap', None)
                })
            
            logger.info(f"Retrieved {len(bars)} bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def get_latest_quotes(self, symbols: List[str] = None) -> Dict:
        """Get latest quotes for symbols"""
        try:
            if not self.rest_client:
                return {}
            
            if not symbols:
                symbols = self.symbols
            
            quotes = {}
            for symbol in symbols:
                try:
                    quote = self.rest_client.get_last_quote(f"C:{symbol}")
                    if quote:
                        quotes[symbol] = {
                            'bid': quote.bid,
                            'ask': quote.ask,
                            'timestamp': quote.timestamp,
                            'spread': quote.ask - quote.bid
                        }
                except Exception as e:
                    logger.warning(f"Failed to get quote for {symbol}: {e}")
                    
            return quotes
            
        except Exception as e:
            logger.error(f"Error getting latest quotes: {e}")
            return {}
    
    def start_websocket_stream(self, callback: Callable):
        """Start real-time WebSocket stream"""
        if not self.api_key:
            logger.error("Cannot start WebSocket stream: API key not configured")
            return
        
        self.callback = callback
        self.is_streaming = True
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if isinstance(data, list):
                    for item in data:
                        self._process_websocket_message(item)
                else:
                    self._process_websocket_message(data)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
            self.is_streaming = False
            # Attempt to reconnect after 5 seconds
            if self.is_streaming:
                time.sleep(5)
                self.start_websocket_stream(callback)
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            ws.send(json.dumps(auth_message))
            
            # Subscribe to forex data
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join([f"C.{symbol}" for symbol in self.symbols])
            }
            ws.send(json.dumps(subscribe_message))
        
        # Create WebSocket connection
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/forex",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run in separate thread
        def run_websocket():
            self.ws.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        logger.info("WebSocket stream started")
    
    def _process_websocket_message(self, message: Dict):
        """Process individual WebSocket message"""
        try:
            if message.get('ev') == 'C':  # Currency quote
                symbol = message.get('p', '').replace('C:', '')
                if symbol in self.symbols:
                    market_data = {
                        'symbol': symbol,
                        'bid': message.get('b'),
                        'ask': message.get('a'),
                        'timestamp': message.get('t'),
                        'type': 'quote'
                    }
                    
                    if self.callback:
                        self.callback(market_data)
                        
            elif message.get('ev') == 'CA':  # Currency aggregate (minute bar)
                symbol = message.get('pair', '').replace('C:', '')
                if symbol in self.symbols:
                    market_data = {
                        'symbol': symbol,
                        'open': message.get('o'),
                        'high': message.get('h'),
                        'low': message.get('l'),
                        'close': message.get('c'),
                        'volume': message.get('v'),
                        'timestamp': message.get('s'),  # Start time
                        'type': 'bar'
                    }
                    
                    if self.callback:
                        self.callback(market_data)
                        
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def stop_stream(self):
        """Stop the WebSocket stream"""
        self.is_streaming = False
        if self.ws:
            self.ws.close()
        logger.info("WebSocket stream stopped")
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            if not self.rest_client:
                return {'status': 'unknown', 'message': 'Client not configured'}
            
            # Forex market is open 24/5, check if it's weekend
            now = datetime.now()
            if now.weekday() >= 5:  # Saturday or Sunday
                return {
                    'status': 'closed',
                    'message': 'Forex market is closed on weekends'
                }
            else:
                return {
                    'status': 'open',
                    'message': 'Forex market is open'
                }
                
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def test_connection(self) -> bool:
        """Test the Polygon.io connection"""
        try:
            if not self.rest_client:
                return False
            
            # Try to get a simple quote
            quote = self.rest_client.get_last_quote("C:EURUSD")
            return quote is not None
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
