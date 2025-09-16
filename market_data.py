"""
Market Data Integration and Backtesting Framework
The Quant Compass - AI Trading Platform

Integrates with Polygon.io for real-time and historical market data,
provides backtesting capabilities for SMC strategies.
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported timeframes"""
    M1 = "1/minute"
    M5 = "5/minute"
    M15 = "15/minute"
    M30 = "30/minute"
    H1 = "1/hour"
    H4 = "4/hour"
    D1 = "1/day"
    W1 = "1/week"


class AssetType(Enum):
    """Supported asset types"""
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    INDICES = "indices"
    COMMODITIES = "commodities"


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    asset_type: AssetType


@dataclass
class BacktestResult:
    """Backtesting result structure"""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    trades: List[Dict]
    equity_curve: pd.DataFrame


class PolygonDataProvider:
    """
    Polygon.io data provider for real-time and historical market data
    
    Supports:
    - Forex majors and crosses
    - Cryptocurrencies
    - Stock indices
    - Commodities (metals, energy)
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Symbol mappings for different asset types
        self.forex_symbols = {
            # Major pairs
            'EURUSD': 'C:EURUSD',
            'GBPUSD': 'C:GBPUSD',
            'USDJPY': 'C:USDJPY',
            'USDCHF': 'C:USDCHF',
            'AUDUSD': 'C:AUDUSD',
            'USDCAD': 'C:USDCAD',
            'NZDUSD': 'C:NZDUSD',
            
            # Cross pairs
            'EURJPY': 'C:EURJPY',
            'GBPJPY': 'C:GBPJPY',
            'EURGBP': 'C:EURGBP',
            'EURAUD': 'C:EURAUD',
            'EURCHF': 'C:EURCHF',
            'AUDJPY': 'C:AUDJPY',
            'CHFJPY': 'C:CHFJPY',
            'GBPAUD': 'C:GBPAUD',
            'GBPCHF': 'C:GBPCHF',
            'AUDCAD': 'C:AUDCAD',
            'AUDCHF': 'C:AUDCHF',
            'AUDNZD': 'C:AUDNZD',
            'CADCHF': 'C:CADCHF',
            'CADJPY': 'C:CADJPY',
            'EURAUD': 'C:EURAUD',
            'EURCAD': 'C:EURCAD',
            'EURNZD': 'C:EURNZD',
            'GBPCAD': 'C:GBPCAD',
            'GBPNZD': 'C:GBPNZD',
            'NZDCAD': 'C:NZDCAD',
            'NZDCHF': 'C:NZDCHF',
            'NZDJPY': 'C:NZDJPY'
        }
        
        self.crypto_symbols = {
            'BTCUSD': 'X:BTCUSD',
            'ETHUSD': 'X:ETHUSD',
            'ADAUSD': 'X:ADAUSD',
            'DOTUSD': 'X:DOTUSD',
            'LINKUSD': 'X:LINKUSD',
            'LTCUSD': 'X:LTCUSD',
            'XRPUSD': 'X:XRPUSD',
            'SOLUSD': 'X:SOLUSD',
            'AVAXUSD': 'X:AVAXUSD',
            'MATICUSD': 'X:MATICUSD'
        }
        
        self.commodity_symbols = {
            # Metals
            'XAUUSD': 'C:XAUUSD',  # Gold
            'XAGUSD': 'C:XAGUSD',  # Silver
            'XPTUSD': 'C:XPTUSD',  # Platinum
            'XPDUSD': 'C:XPDUSD',  # Palladium
            
            # Energy
            'USOIL': 'C:USOIL',    # Crude Oil
            'UKOIL': 'C:UKOIL',    # Brent Oil
            'NGAS': 'C:NGAS'       # Natural Gas
        }
        
        self.index_symbols = {
            'SPX': 'I:SPX',        # S&P 500
            'DJI': 'I:DJI',        # Dow Jones
            'IXIC': 'I:IXIC',      # NASDAQ
            'RUT': 'I:RUT',        # Russell 2000
            'VIX': 'I:VIX',        # VIX
            'DAX': 'I:DAX',        # German DAX
            'FTSE': 'I:FTSE',      # FTSE 100
            'N225': 'I:N225',      # Nikkei 225
            'HSI': 'I:HSI'         # Hang Seng
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_polygon_symbol(self, symbol: str) -> str:
        """Convert symbol to Polygon.io format"""
        # Check all symbol mappings
        for symbol_dict in [self.forex_symbols, self.crypto_symbols, 
                           self.commodity_symbols, self.index_symbols]:
            if symbol in symbol_dict:
                return symbol_dict[symbol]
        
        # If not found, return as-is (might be already in Polygon format)
        return symbol
    
    def _get_asset_type(self, symbol: str) -> AssetType:
        """Determine asset type from symbol"""
        if symbol in self.forex_symbols:
            return AssetType.FOREX
        elif symbol in self.crypto_symbols:
            return AssetType.CRYPTO
        elif symbol in self.commodity_symbols:
            return AssetType.COMMODITIES
        elif symbol in self.index_symbols:
            return AssetType.INDICES
        else:
            return AssetType.FOREX  # Default to forex
    
    async def get_historical_data(self, symbol: str, timeframe: Timeframe, 
                                start_date: datetime, end_date: datetime,
                                limit: int = 5000) -> pd.DataFrame:
        """
        Get historical OHLCV data from Polygon.io
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        polygon_symbol = self._get_polygon_symbol(symbol)
        asset_type = self._get_asset_type(symbol)
        
        # Format dates for Polygon API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Build API URL based on asset type
        if asset_type == AssetType.FOREX:
            endpoint = f"/v2/aggs/ticker/{polygon_symbol}/range/{timeframe.value}/{start_str}/{end_str}"
        elif asset_type == AssetType.CRYPTO:
            endpoint = f"/v2/aggs/ticker/{polygon_symbol}/range/{timeframe.value}/{start_str}/{end_str}"
        else:
            endpoint = f"/v2/aggs/ticker/{polygon_symbol}/range/{timeframe.value}/{start_str}/{end_str}"
        
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': limit
        }
        
        url = f"{self.base_url}{endpoint}?" + urlencode(params)
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'OK' and 'results' in data:
                        df = self._parse_polygon_data(data['results'], symbol, timeframe.value, asset_type)
                        logger.info(f"Retrieved {len(df)} candles for {symbol} ({timeframe.value})")
                        return df
                    else:
                        logger.error(f"Polygon API error: {data.get('error', 'Unknown error')}")
                        return pd.DataFrame()
                else:
                    logger.error(f"HTTP error {response.status} for {symbol}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _parse_polygon_data(self, results: List[Dict], symbol: str, 
                          timeframe: str, asset_type: AssetType) -> pd.DataFrame:
        """Parse Polygon.io API response into DataFrame"""
        
        data = []
        for candle in results:
            data.append({
                'timestamp': pd.to_datetime(candle['t'], unit='ms'),
                'open': candle['o'],
                'high': candle['h'],
                'low': candle['l'],
                'close': candle['c'],
                'volume': candle.get('v', 0),  # Some assets might not have volume
                'symbol': symbol,
                'timeframe': timeframe,
                'asset_type': asset_type.value
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    async def get_real_time_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a symbol"""
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        polygon_symbol = self._get_polygon_symbol(symbol)
        asset_type = self._get_asset_type(symbol)
        
        # Use appropriate endpoint based on asset type
        if asset_type == AssetType.FOREX:
            endpoint = f"/v1/last_quote/currencies/{polygon_symbol}"
        elif asset_type == AssetType.CRYPTO:
            endpoint = f"/v1/last/crypto/{polygon_symbol}"
        else:
            endpoint = f"/v2/last/trade/{polygon_symbol}"
        
        params = {'apikey': self.api_key}
        url = f"{self.base_url}{endpoint}?" + urlencode(params)
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'OK':
                        return self._parse_quote_data(data, symbol, asset_type)
                    else:
                        logger.error(f"Quote API error for {symbol}: {data.get('error')}")
                        return None
                else:
                    logger.error(f"HTTP error {response.status} for quote {symbol}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return None
    
    def _parse_quote_data(self, data: Dict, symbol: str, asset_type: AssetType) -> Dict:
        """Parse quote data from Polygon API"""
        
        if asset_type == AssetType.FOREX and 'last' in data:
            quote_data = data['last']
            return {
                'symbol': symbol,
                'bid': quote_data.get('bid'),
                'ask': quote_data.get('ask'),
                'timestamp': pd.to_datetime(quote_data.get('timestamp'), unit='ms'),
                'spread': quote_data.get('ask', 0) - quote_data.get('bid', 0)
            }
        elif 'results' in data:
            quote_data = data['results']
            price = quote_data.get('p', quote_data.get('price', 0))
            return {
                'symbol': symbol,
                'price': price,
                'timestamp': pd.to_datetime(quote_data.get('t'), unit='ms'),
                'volume': quote_data.get('s', quote_data.get('size', 0))
            }
        
        return {'symbol': symbol, 'error': 'Unable to parse quote data'}


class BacktestEngine:
    """
    Comprehensive backtesting engine for SMC strategies
    
    Features:
    - Historical data simulation
    - Realistic spread and slippage modeling
    - Commission calculation
    - Risk management integration
    - Performance analytics
    """
    
    def __init__(self, data_provider: PolygonDataProvider, config: Dict = None):
        self.data_provider = data_provider
        self.config = config or self._default_config()
        self.results: Optional[BacktestResult] = None
        
    def _default_config(self) -> Dict:
        """Default backtesting configuration"""
        return {
            'initial_balance': 100000.0,
            'commission_per_lot': 7.0,      # $7 per round trip
            'spread_pips': {
                'EURUSD': 0.8, 'GBPUSD': 1.2, 'USDJPY': 0.9,
                'USDCHF': 1.1, 'AUDUSD': 1.0, 'USDCAD': 1.3,
                'NZDUSD': 1.5, 'XAUUSD': 3.0, 'BTCUSD': 10.0
            },
            'slippage_pips': 0.5,           # Average slippage
            'max_spread_pips': 5.0,         # Skip trades if spread too wide
            'weekend_trading': False,       # Skip weekend periods
            'news_filter': False,           # Filter high-impact news times
            'max_concurrent_trades': 5,     # Maximum open positions
            'margin_requirement': 0.02      # 2% margin (50:1 leverage)
        }
    
    async def run_backtest(self, symbol: str, timeframe: Timeframe,
                          start_date: datetime, end_date: datetime,
                          strategy_func, risk_manager) -> BacktestResult:
        """
        Run comprehensive backtest
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Backtest start date
            end_date: Backtest end date
            strategy_func: Strategy function that generates signals
            risk_manager: Risk management instance
            
        Returns:
            BacktestResult with complete performance metrics
        """
        
        logger.info(f"Starting backtest: {symbol} {timeframe.value} from {start_date} to {end_date}")
        
        # Get historical data
        df = await self.data_provider.get_historical_data(
            symbol, timeframe, start_date, end_date
        )
        
        if df.empty:
            raise ValueError(f"No data available for {symbol} in specified period")
        
        # Initialize backtest state
        balance = self.config['initial_balance']
        equity = balance
        trades = []
        open_positions = []
        equity_curve = []
        
        # Update risk manager with initial balance
        risk_manager.update_account_metrics(balance, equity)
        
        # Process each candle
        for i in range(len(df)):
            current_candle = df.iloc[i:i+1]
            current_price = current_candle['close'].iloc[0]
            current_time = current_candle['timestamp'].iloc[0]
            
            # Skip weekends if configured
            if not self.config['weekend_trading'] and current_time.weekday() >= 5:
                continue
            
            # Update open positions
            balance, equity, closed_trades = self._update_positions(
                open_positions, current_candle, balance, symbol
            )
            trades.extend(closed_trades)
            
            # Update risk manager
            daily_pnl = equity - self.config['initial_balance']
            risk_manager.update_account_metrics(balance, equity, daily_pnl=daily_pnl)
            
            # Check for emergency stop
            emergency = risk_manager.check_emergency_stop()
            if emergency['emergency_stop']:
                logger.warning(f"Emergency stop triggered: {emergency['conditions']}")
                break
            
            # Generate trading signal if we have enough data
            if i >= 50:  # Need sufficient history for SMC analysis
                historical_data = df.iloc[:i+1].copy()
                
                try:
                    # Get strategy signal
                    signal = strategy_func(historical_data)
                    
                    if signal and len(open_positions) < self.config['max_concurrent_trades']:
                        # Assess trade risk
                        trade_risk = risk_manager.assess_trade_risk(
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            confidence=signal.confidence,
                            symbol=symbol
                        )
                        
                        if trade_risk.approved:
                            # Execute trade
                            trade = self._execute_trade(
                                signal, trade_risk, current_time, current_price, symbol
                            )
                            
                            if trade:
                                open_positions.append(trade)
                                balance -= trade['commission']
                                
                                # Record trade with risk manager
                                risk_manager.record_trade({
                                    'symbol': symbol,
                                    'direction': signal.signal_type.value,
                                    'lot_size': trade_risk.position_size.lot_size,
                                    'entry_price': trade['entry_price'],
                                    'stop_loss': trade['stop_loss'],
                                    'take_profit': trade['take_profit'],
                                    'risk_amount': trade_risk.position_size.risk_amount,
                                    'status': 'open'
                                })
                
                except Exception as e:
                    logger.error(f"Strategy error at {current_time}: {str(e)}")
            
            # Record equity curve
            equity_curve.append({
                'timestamp': current_time,
                'balance': balance,
                'equity': equity,
                'drawdown': (self.config['initial_balance'] - equity) / self.config['initial_balance'] * 100
            })
        
        # Close any remaining open positions
        for position in open_positions:
            final_price = df['close'].iloc[-1]
            closed_trade = self._close_position(position, final_price, df['timestamp'].iloc[-1])
            trades.append(closed_trade)
            balance += closed_trade['pnl'] - closed_trade['commission']
        
        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve)
        result = self._calculate_performance_metrics(
            symbol, timeframe.value, start_date, end_date, trades, equity_df
        )
        
        self.results = result
        logger.info(f"Backtest completed: {result.total_trades} trades, {result.win_rate:.1f}% win rate, {result.total_pnl:.2f} total PnL")
        
        return result
    
    def _update_positions(self, open_positions: List[Dict], current_candle: pd.DataFrame,
                         balance: float, symbol: str) -> Tuple[float, float, List[Dict]]:
        """Update open positions and close if stop/target hit"""
        
        current_high = current_candle['high'].iloc[0]
        current_low = current_candle['low'].iloc[0]
        current_time = current_candle['timestamp'].iloc[0]
        
        closed_trades = []
        remaining_positions = []
        
        for position in open_positions:
            closed = False
            
            if position['direction'] == 'buy':
                # Check stop loss
                if current_low <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    closed_trade = self._close_position(position, exit_price, current_time, 'stop_loss')
                    closed_trades.append(closed_trade)
                    balance += closed_trade['pnl'] - closed_trade['commission']
                    closed = True
                
                # Check take profit
                elif current_high >= position['take_profit']:
                    exit_price = position['take_profit']
                    closed_trade = self._close_position(position, exit_price, current_time, 'take_profit')
                    closed_trades.append(closed_trade)
                    balance += closed_trade['pnl'] - closed_trade['commission']
                    closed = True
            
            else:  # sell position
                # Check stop loss
                if current_high >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    closed_trade = self._close_position(position, exit_price, current_time, 'stop_loss')
                    closed_trades.append(closed_trade)
                    balance += closed_trade['pnl'] - closed_trade['commission']
                    closed = True
                
                # Check take profit
                elif current_low <= position['take_profit']:
                    exit_price = position['take_profit']
                    closed_trade = self._close_position(position, exit_price, current_time, 'take_profit')
                    closed_trades.append(closed_trade)
                    balance += closed_trade['pnl'] - closed_trade['commission']
                    closed = True
            
            if not closed:
                remaining_positions.append(position)
        
        # Update open positions list
        open_positions.clear()
        open_positions.extend(remaining_positions)
        
        # Calculate current equity (including floating P&L)
        equity = balance
        for position in open_positions:
            current_price = (current_high + current_low) / 2  # Mid price
            floating_pnl = self._calculate_pnl(position, current_price)
            equity += floating_pnl
        
        return balance, equity, closed_trades
    
    def _execute_trade(self, signal, trade_risk, timestamp: datetime, 
                      current_price: float, symbol: str) -> Optional[Dict]:
        """Execute a trade based on signal and risk assessment"""
        
        # Apply spread and slippage
        spread = self.config['spread_pips'].get(symbol, 1.0) / 10000  # Convert pips to price
        slippage = self.config['slippage_pips'] / 10000
        
        if signal.signal_type.value == 'buy':
            entry_price = current_price + spread/2 + slippage  # Buy at ask + slippage
        else:
            entry_price = current_price - spread/2 - slippage  # Sell at bid - slippage
        
        # Check if spread is acceptable
        if spread > self.config['max_spread_pips'] / 10000:
            logger.warning(f"Spread too wide for {symbol}: {spread*10000:.1f} pips")
            return None
        
        # Calculate commission
        commission = self.config['commission_per_lot'] * trade_risk.position_size.lot_size
        
        trade = {
            'id': f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'symbol': symbol,
            'direction': signal.signal_type.value,
            'entry_price': entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'lot_size': trade_risk.position_size.lot_size,
            'entry_time': timestamp,
            'commission': commission,
            'confidence': signal.confidence,
            'risk_reward_ratio': trade_risk.risk_reward_ratio
        }
        
        return trade
    
    def _close_position(self, position: Dict, exit_price: float, 
                       exit_time: datetime, exit_reason: str = 'manual') -> Dict:
        """Close a position and calculate P&L"""
        
        pnl = self._calculate_pnl(position, exit_price)
        commission = self.config['commission_per_lot'] * position['lot_size']
        
        closed_trade = {
            **position,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'commission': commission,
            'net_pnl': pnl - commission,
            'duration': exit_time - position['entry_time']
        }
        
        return closed_trade
    
    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate P&L for a position"""
        
        pip_value = 10.0  # $10 per pip for standard lot (simplified)
        price_diff = current_price - position['entry_price']
        
        if position['direction'] == 'buy':
            pnl = price_diff * pip_value * position['lot_size'] * 10000
        else:  # sell
            pnl = -price_diff * pip_value * position['lot_size'] * 10000
        
        return pnl
    
    def _calculate_performance_metrics(self, symbol: str, timeframe: str,
                                     start_date: datetime, end_date: datetime,
                                     trades: List[Dict], equity_curve: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return BacktestResult(
                symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date,
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                total_pnl=0.0, max_drawdown=0.0, sharpe_ratio=0.0, profit_factor=0.0,
                avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
                trades=[], equity_curve=equity_curve
            )
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['net_pnl'] > 0])
        losing_trades = len([t for t in trades if t['net_pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L statistics
        pnls = [t['net_pnl'] for t in trades]
        total_pnl = sum(pnls)
        
        winning_pnls = [t['net_pnl'] for t in trades if t['net_pnl'] > 0]
        losing_pnls = [t['net_pnl'] for t in trades if t['net_pnl'] < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        if not equity_curve.empty:
            equity_curve['peak'] = equity_curve['equity'].cummax()
            equity_curve['drawdown_pct'] = (equity_curve['peak'] - equity_curve['equity']) / equity_curve['peak'] * 100
            max_drawdown = equity_curve['drawdown_pct'].max()
        else:
            max_drawdown = 0.0
        
        # Sharpe ratio (simplified)
        if not equity_curve.empty and len(equity_curve) > 1:
            returns = equity_curve['equity'].pct_change().dropna()
            if returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def get_performance_summary(self) -> Dict:
        """Get formatted performance summary"""
        if not self.results:
            return {'error': 'No backtest results available'}
        
        r = self.results
        
        return {
            'symbol': r.symbol,
            'timeframe': r.timeframe,
            'period': f"{r.start_date.strftime('%Y-%m-%d')} to {r.end_date.strftime('%Y-%m-%d')}",
            'total_trades': r.total_trades,
            'win_rate': f"{r.win_rate:.1f}%",
            'total_pnl': f"${r.total_pnl:,.2f}",
            'max_drawdown': f"{r.max_drawdown:.2f}%",
            'profit_factor': f"{r.profit_factor:.2f}",
            'sharpe_ratio': f"{r.sharpe_ratio:.2f}",
            'avg_win': f"${r.avg_win:.2f}",
            'avg_loss': f"${r.avg_loss:.2f}",
            'largest_win': f"${r.largest_win:.2f}",
            'largest_loss': f"${r.largest_loss:.2f}",
            'risk_reward': f"1:{abs(r.avg_win/r.avg_loss):.2f}" if r.avg_loss != 0 else "N/A"
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    from smc_engine import SMCEngine
    from risk_manager import RiskManager
    
    async def example_backtest():
        # Initialize components
        api_key = os.getenv('POLYGON_API_KEY', 'your_api_key_here')
        
        async with PolygonDataProvider(api_key) as data_provider:
            # Initialize SMC engine and risk manager
            smc_engine = SMCEngine()
            risk_manager = RiskManager()
            
            # Create backtest engine
            backtest_engine = BacktestEngine(data_provider)
            
            # Define strategy function
            def smc_strategy(df: pd.DataFrame):
                analysis = smc_engine.analyze_market_data(df)
                if analysis.get('signal'):
                    return analysis['signal']
                return None
            
            # Run backtest
            try:
                result = await backtest_engine.run_backtest(
                    symbol='EURUSD',
                    timeframe=Timeframe.H1,
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 6, 30),
                    strategy_func=smc_strategy,
                    risk_manager=risk_manager
                )
                
                # Print results
                summary = backtest_engine.get_performance_summary()
                print("Backtest Results:")
                for key, value in summary.items():
                    print(f"{key}: {value}")
                    
            except Exception as e:
                print(f"Backtest error: {e}")
    
    # Run example
    # asyncio.run(example_backtest())
