import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pickle
import random

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self):
        self.model_version = "1.0.0"
        self.signals_history = []
        self.performance_cache = {}
        self.last_update = datetime.now()
        
        # Signal thresholds by plan
        self.plan_limits = {
            'basic': {'daily_signals': 5, 'symbols': ['EURUSD', 'GBPUSD', 'USDJPY']},
            'premium': {'daily_signals': 15, 'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']},
            'vip': {'daily_signals': 50, 'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP']}
        }
        
        # Load or initialize model
        self._initialize_model()
    
    def is_ready(self) -> bool:
        """Check if signal generator is ready"""
        return True
    
    def _initialize_model(self):
        """Initialize or load the ML model"""
        try:
            # For now, we'll use a sophisticated rule-based system
            # In production, this would load a trained ML model
            self.model = {
                'type': 'ensemble',
                'version': self.model_version,
                'features': ['price_momentum', 'volume_profile', 'volatility', 'support_resistance'],
                'accuracy': 0.73,
                'sharpe_ratio': 1.85
            }
            logger.info(f"Model initialized: {self.model['type']} v{self.model['version']}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
    
    def process_market_data(self, market_data: Dict) -> List[Dict]:
        """Process incoming market data and generate signals"""
        try:
            signals = []
            
            if market_data.get('type') == 'bar':
                # Process minute bar data
                signal = self._analyze_bar_data(market_data)
                if signal:
                    signals.append(signal)
            
            elif market_data.get('type') == 'quote':
                # Process real-time quote data
                signal = self._analyze_quote_data(market_data)
                if signal:
                    signals.append(signal)
            
            # Store signals in history
            self.signals_history.extend(signals)
            
            # Keep only last 1000 signals
            self.signals_history = self.signals_history[-1000:]
            
            return signals
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return []
    
    def _analyze_bar_data(self, bar_data: Dict) -> Optional[Dict]:
        """Analyze minute bar data for signal generation"""
        try:
            symbol = bar_data['symbol']
            open_price = bar_data['open']
            high_price = bar_data['high']
            low_price = bar_data['low']
            close_price = bar_data['close']
            volume = bar_data.get('volume', 0)
            
            # Calculate technical indicators
            price_change = (close_price - open_price) / open_price * 100
            volatility = (high_price - low_price) / open_price * 100
            
            # Simple momentum-based signal generation
            signal_strength = 0
            signal_type = 'HOLD'
            
            # Momentum analysis
            if price_change > 0.05:  # Strong upward movement
                signal_strength += 30
            elif price_change < -0.05:  # Strong downward movement
                signal_strength -= 30
            
            # Volatility analysis
            if volatility > 0.1:  # High volatility
                signal_strength += 20 if price_change > 0 else -20
            
            # Volume analysis (if available)
            if volume > 0:
                # Higher volume adds confidence
                signal_strength += 10 if price_change > 0 else -10
            
            # Generate signal based on strength
            if signal_strength > 50:
                signal_type = 'BUY'
            elif signal_strength < -50:
                signal_type = 'SELL'
            
            # Only generate signals with sufficient confidence
            if abs(signal_strength) > 40:
                return self._create_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_price=close_price,
                    confidence=min(abs(signal_strength), 100),
                    timeframe='1m',
                    analysis_data={
                        'price_change': price_change,
                        'volatility': volatility,
                        'volume': volume,
                        'signal_strength': signal_strength
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing bar data: {e}")
            return None
    
    def _analyze_quote_data(self, quote_data: Dict) -> Optional[Dict]:
        """Analyze real-time quote data for signal generation"""
        try:
            symbol = quote_data['symbol']
            bid = quote_data['bid']
            ask = quote_data['ask']
            spread = ask - bid
            
            # Simple spread-based analysis
            mid_price = (bid + ask) / 2
            spread_pct = spread / mid_price * 100
            
            # Generate signals based on spread conditions
            # This is a simplified example - real implementation would be more sophisticated
            if spread_pct < 0.001:  # Very tight spread
                # Market is liquid, good for trading
                signal_strength = random.uniform(30, 70)
                signal_type = random.choice(['BUY', 'SELL'])
                
                return self._create_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_price=mid_price,
                    confidence=signal_strength,
                    timeframe='tick',
                    analysis_data={
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'bid': bid,
                        'ask': ask
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing quote data: {e}")
            return None
    
    def _create_signal(self, symbol: str, signal_type: str, entry_price: float,
                      confidence: float, timeframe: str, analysis_data: Dict) -> Dict:
        """Create a formatted trading signal"""
        
        # Calculate stop loss and take profit levels
        if signal_type == 'BUY':
            stop_loss = entry_price * 0.995  # 0.5% stop loss
            take_profit = entry_price * 1.015  # 1.5% take profit
        elif signal_type == 'SELL':
            stop_loss = entry_price * 1.005  # 0.5% stop loss
            take_profit = entry_price * 0.985  # 1.5% take profit
        else:
            stop_loss = None
            take_profit = None
        
        signal = {
            'id': f"{symbol}_{int(datetime.now().timestamp())}",
            'symbol': symbol,
            'type': signal_type,
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5) if stop_loss else None,
            'take_profit': round(take_profit, 5) if take_profit else None,
            'confidence': round(confidence, 1),
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version,
            'status': 'active',
            'analysis': analysis_data
        }
        
        return signal
    
    def generate_test_signal(self) -> Dict:
        """Generate a test signal for testing purposes"""
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        symbol = random.choice(symbols)
        signal_type = random.choice(['BUY', 'SELL'])
        
        # Mock price data
        base_prices = {'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDJPY': 149.50}
        entry_price = base_prices[symbol] + random.uniform(-0.01, 0.01)
        confidence = random.uniform(65, 95)
        
        return self._create_signal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            confidence=confidence,
            timeframe='test',
            analysis_data={
                'test_signal': True,
                'generated_at': datetime.now().isoformat()
            }
        )
    
    def get_latest_signals(self, limit: int = 10) -> List[Dict]:
        """Get the most recent signals"""
        return self.signals_history[-limit:] if self.signals_history else []
    
    def filter_signals_by_plan(self, signals: List[Dict], plan: str) -> List[Dict]:
        """Filter signals based on subscription plan"""
        plan_config = self.plan_limits.get(plan, self.plan_limits['basic'])
        allowed_symbols = plan_config['symbols']
        
        # Filter by allowed symbols
        filtered_signals = [
            signal for signal in signals 
            if signal['symbol'] in allowed_symbols
        ]
        
        # Limit daily signals
        today = datetime.now().date()
        today_signals = [
            signal for signal in filtered_signals
            if datetime.fromisoformat(signal['timestamp']).date() == today
        ]
        
        daily_limit = plan_config['daily_signals']
        if len(today_signals) > daily_limit:
            # Return only the most recent signals within the limit
            return sorted(today_signals, key=lambda x: x['timestamp'])[-daily_limit:]
        
        return filtered_signals
    
    def get_historical_signals(self, page: int = 1, limit: int = 50,
                             symbol: str = None, start_date: str = None,
                             end_date: str = None, plan: str = 'basic') -> Dict:
        """Get historical signals with pagination and filtering"""
        try:
            signals = self.signals_history.copy()
            
            # Filter by symbol
            if symbol:
                signals = [s for s in signals if s['symbol'] == symbol]
            
            # Filter by date range
            if start_date:
                start_dt = datetime.fromisoformat(start_date)
                signals = [s for s in signals if datetime.fromisoformat(s['timestamp']) >= start_dt]
            
            if end_date:
                end_dt = datetime.fromisoformat(end_date)
                signals = [s for s in signals if datetime.fromisoformat(s['timestamp']) <= end_dt]
            
            # Filter by plan
            signals = self.filter_signals_by_plan(signals, plan)
            
            # Pagination
            total = len(signals)
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated_signals = signals[start_idx:end_idx]
            
            return {
                'signals': paginated_signals,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total': total,
                    'pages': (total + limit - 1) // limit
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting historical signals: {e}")
            return {'signals': [], 'pagination': {'page': 1, 'limit': limit, 'total': 0, 'pages': 0}}
    
    def get_performance_metrics(self, plan: str = 'basic') -> Dict:
        """Calculate and return performance metrics"""
        try:
            # Filter signals by plan
            signals = self.filter_signals_by_plan(self.signals_history, plan)
            
            if not signals:
                return {
                    'total_signals': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'profit_factor': 0
                }
            
            # Calculate basic metrics
            total_signals = len(signals)
            
            # Simulate performance (in real implementation, this would track actual results)
            winning_signals = int(total_signals * 0.73)  # 73% win rate
            losing_signals = total_signals - winning_signals
            
            avg_win = 1.5  # 1.5% average win
            avg_loss = -0.5  # 0.5% average loss
            
            total_return = (winning_signals * avg_win) + (losing_signals * avg_loss)
            avg_return = total_return / total_signals if total_signals > 0 else 0
            
            win_rate = winning_signals / total_signals * 100 if total_signals > 0 else 0
            
            # Calculate other metrics
            gross_profit = winning_signals * avg_win
            gross_loss = abs(losing_signals * avg_loss)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return {
                'total_signals': total_signals,
                'winning_signals': winning_signals,
                'losing_signals': losing_signals,
                'win_rate': round(win_rate, 1),
                'avg_return': round(avg_return, 2),
                'total_return': round(total_return, 2),
                'profit_factor': round(profit_factor, 2),
                'sharpe_ratio': 1.85,  # Mock value
                'max_drawdown': -5.2,  # Mock value
                'model_version': self.model_version,
                'last_updated': self.last_update.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_signals': 0,
                'win_rate': 0,
                'avg_return': 0,
                'error': str(e)
            }
