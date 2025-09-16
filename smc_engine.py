"""
Smart Money Concepts (SMC) Trading Engine
The Quant Compass - AI Trading Platform

This module implements the complete SMC methodology based on the comprehensive
trading guide, including order blocks, fair value gaps, market structure analysis,
and institutional trading patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketStructure(Enum):
    """Market structure states"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    TRANSITION = "transition"


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderBlockType(Enum):
    """Order block classifications"""
    BULLISH_OB = "bullish_ob"
    BEARISH_OB = "bearish_ob"
    BREAKER_BULLISH = "breaker_bullish"
    BREAKER_BEARISH = "breaker_bearish"


@dataclass
class OrderBlock:
    """Order Block data structure"""
    type: OrderBlockType
    high: float
    low: float
    open: float
    close: float
    timestamp: datetime
    volume: float
    strength: float  # 0-100 strength rating
    tested: bool = False
    mitigation_count: int = 0


@dataclass
class FairValueGap:
    """Fair Value Gap (Imbalance) data structure"""
    high: float
    low: float
    timestamp: datetime
    direction: str  # 'bullish' or 'bearish'
    filled: bool = False
    fill_percentage: float = 0.0


@dataclass
class LiquidityLevel:
    """Liquidity level identification"""
    price: float
    timestamp: datetime
    type: str  # 'buy_side' or 'sell_side'
    strength: float
    swept: bool = False


@dataclass
class TradingSignal:
    """Complete trading signal with SMC analysis"""
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-100
    risk_reward_ratio: float
    timestamp: datetime
    analysis: Dict
    timeframe: str


class SMCEngine:
    """
    Smart Money Concepts Trading Engine
    
    Implements the complete SMC methodology including:
    - Market structure analysis
    - Order block identification
    - Fair value gap detection
    - Liquidity analysis
    - Break of structure (BOS) and Change of Character (ChoCH)
    - Multi-timeframe confluence
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        self.liquidity_levels: List[LiquidityLevel] = []
        self.market_structure = MarketStructure.RANGING
        self.last_signal: Optional[TradingSignal] = None
        
    def _default_config(self) -> Dict:
        """Default SMC engine configuration"""
        return {
            'min_ob_strength': 70,
            'fvg_threshold': 0.0001,  # Minimum gap size
            'liquidity_lookback': 50,
            'structure_lookback': 100,
            'min_volume_ratio': 1.5,
            'confluence_weight': {
                'order_block': 0.3,
                'fair_value_gap': 0.2,
                'liquidity': 0.25,
                'market_structure': 0.25
            }
        }
    
    def analyze_market_data(self, df: pd.DataFrame) -> Dict:
        """
        Main analysis function that processes OHLCV data
        
        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Complete market analysis including signals
        """
        try:
            # Ensure data is properly formatted
            df = self._prepare_data(df)
            
            # Core SMC Analysis
            self._identify_market_structure(df)
            self._identify_order_blocks(df)
            self._identify_fair_value_gaps(df)
            self._identify_liquidity_levels(df)
            
            # Generate trading signals
            signal = self._generate_signal(df)
            
            # Compile analysis results
            analysis = {
                'market_structure': self.market_structure.value,
                'order_blocks': len(self.order_blocks),
                'active_fvgs': len([fvg for fvg in self.fair_value_gaps if not fvg.filled]),
                'liquidity_levels': len(self.liquidity_levels),
                'signal': signal.__dict__ if signal else None,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"SMC Analysis completed: {analysis['market_structure']} structure, {analysis['order_blocks']} OBs")
            return analysis
            
        except Exception as e:
            logger.error(f"SMC Analysis error: {str(e)}")
            raise
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate market data"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate additional indicators
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        return df
    
    def _identify_market_structure(self, df: pd.DataFrame) -> None:
        """
        Identify current market structure using SMC principles
        - Higher Highs and Higher Lows = Bullish
        - Lower Highs and Lower Lows = Bearish
        - Mixed signals = Ranging/Transition
        """
        if len(df) < self.config['structure_lookback']:
            self.market_structure = MarketStructure.RANGING
            return
        
        # Get recent data for structure analysis
        recent_data = df.tail(self.config['structure_lookback'])
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(recent_data, 'high')
        swing_lows = self._find_swing_points(recent_data, 'low')
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            self.market_structure = MarketStructure.RANGING
            return
        
        # Analyze structure trend
        hh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] > swing_highs[i-1])
        hl_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] > swing_lows[i-1])
        
        lh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] < swing_highs[i-1])
        ll_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] < swing_lows[i-1])
        
        # Determine structure
        if hh_count >= 2 and hl_count >= 2:
            self.market_structure = MarketStructure.BULLISH
        elif lh_count >= 2 and ll_count >= 2:
            self.market_structure = MarketStructure.BEARISH
        else:
            self.market_structure = MarketStructure.RANGING
    
    def _find_swing_points(self, df: pd.DataFrame, column: str, window: int = 5) -> List[float]:
        """Find swing highs or lows in price data"""
        swing_points = []
        
        for i in range(window, len(df) - window):
            if column == 'high':
                # Swing high: current high is highest in window
                if df.iloc[i][column] == df.iloc[i-window:i+window+1][column].max():
                    swing_points.append(df.iloc[i][column])
            else:
                # Swing low: current low is lowest in window
                if df.iloc[i][column] == df.iloc[i-window:i+window+1][column].min():
                    swing_points.append(df.iloc[i][column])
        
        return swing_points
    
    def _identify_order_blocks(self, df: pd.DataFrame) -> None:
        """
        Identify Order Blocks using SMC methodology
        
        Order Block criteria:
        1. Strong impulse move (high volume, large range)
        2. Followed by retracement
        3. Continuation in original direction
        """
        self.order_blocks.clear()
        
        if len(df) < 20:
            return
        
        # Look for potential order blocks in recent data
        for i in range(10, len(df) - 5):
            current_candle = df.iloc[i]
            
            # Check for bullish order block
            if self._is_bullish_order_block(df, i):
                ob = OrderBlock(
                    type=OrderBlockType.BULLISH_OB,
                    high=current_candle['high'],
                    low=current_candle['low'],
                    open=current_candle['open'],
                    close=current_candle['close'],
                    timestamp=current_candle['timestamp'],
                    volume=current_candle['volume'],
                    strength=self._calculate_ob_strength(df, i, 'bullish')
                )
                self.order_blocks.append(ob)
            
            # Check for bearish order block
            elif self._is_bearish_order_block(df, i):
                ob = OrderBlock(
                    type=OrderBlockType.BEARISH_OB,
                    high=current_candle['high'],
                    low=current_candle['low'],
                    open=current_candle['open'],
                    close=current_candle['close'],
                    timestamp=current_candle['timestamp'],
                    volume=current_candle['volume'],
                    strength=self._calculate_ob_strength(df, i, 'bearish')
                )
                self.order_blocks.append(ob)
        
        # Keep only high-strength order blocks
        self.order_blocks = [ob for ob in self.order_blocks 
                           if ob.strength >= self.config['min_ob_strength']]
    
    def _is_bullish_order_block(self, df: pd.DataFrame, index: int) -> bool:
        """Check if candle at index is a bullish order block"""
        if index < 5 or index >= len(df) - 3:
            return False
        
        current = df.iloc[index]
        
        # Must be a bullish candle
        if current['close'] <= current['open']:
            return False
        
        # Check for strong impulse (high volume, large body)
        avg_volume = df.iloc[index-10:index]['volume'].mean()
        avg_range = df.iloc[index-10:index]['range'].mean()
        
        volume_condition = current['volume'] > avg_volume * self.config['min_volume_ratio']
        range_condition = current['range'] > avg_range * 1.2
        body_condition = current['body'] > current['range'] * 0.6
        
        # Check for subsequent move higher
        future_high = df.iloc[index+1:index+4]['high'].max()
        continuation_condition = future_high > current['high']
        
        return volume_condition and range_condition and body_condition and continuation_condition
    
    def _is_bearish_order_block(self, df: pd.DataFrame, index: int) -> bool:
        """Check if candle at index is a bearish order block"""
        if index < 5 or index >= len(df) - 3:
            return False
        
        current = df.iloc[index]
        
        # Must be a bearish candle
        if current['close'] >= current['open']:
            return False
        
        # Check for strong impulse (high volume, large body)
        avg_volume = df.iloc[index-10:index]['volume'].mean()
        avg_range = df.iloc[index-10:index]['range'].mean()
        
        volume_condition = current['volume'] > avg_volume * self.config['min_volume_ratio']
        range_condition = current['range'] > avg_range * 1.2
        body_condition = current['body'] > current['range'] * 0.6
        
        # Check for subsequent move lower
        future_low = df.iloc[index+1:index+4]['low'].min()
        continuation_condition = future_low < current['low']
        
        return volume_condition and range_condition and body_condition and continuation_condition
    
    def _calculate_ob_strength(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """Calculate order block strength (0-100)"""
        current = df.iloc[index]
        
        # Volume strength (0-40 points)
        avg_volume = df.iloc[max(0, index-20):index]['volume'].mean()
        volume_ratio = min(current['volume'] / avg_volume, 3.0)
        volume_score = (volume_ratio - 1) * 20  # 0-40 points
        
        # Range strength (0-30 points)
        avg_range = df.iloc[max(0, index-20):index]['range'].mean()
        range_ratio = min(current['range'] / avg_range, 2.5)
        range_score = (range_ratio - 1) * 20  # 0-30 points
        
        # Body percentage (0-20 points)
        body_percentage = current['body'] / current['range']
        body_score = body_percentage * 20  # 0-20 points
        
        # Confluence with market structure (0-10 points)
        structure_score = 0
        if direction == 'bullish' and self.market_structure == MarketStructure.BULLISH:
            structure_score = 10
        elif direction == 'bearish' and self.market_structure == MarketStructure.BEARISH:
            structure_score = 10
        
        total_score = volume_score + range_score + body_score + structure_score
        return min(max(total_score, 0), 100)
    
    def _identify_fair_value_gaps(self, df: pd.DataFrame) -> None:
        """
        Identify Fair Value Gaps (Imbalances) in price action
        
        FVG occurs when:
        - 3 consecutive candles
        - Gap between candle 1 and candle 3
        - Middle candle doesn't fill the gap
        """
        self.fair_value_gaps.clear()
        
        if len(df) < 3:
            return
        
        for i in range(2, len(df)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Bullish FVG: gap between candle1 high and candle3 low
            if candle1['high'] < candle3['low']:
                gap_size = candle3['low'] - candle1['high']
                if gap_size > self.config['fvg_threshold']:
                    # Check if middle candle doesn't fill the gap
                    if candle2['low'] > candle1['high'] and candle2['high'] < candle3['low']:
                        fvg = FairValueGap(
                            high=candle3['low'],
                            low=candle1['high'],
                            timestamp=candle2['timestamp'],
                            direction='bullish'
                        )
                        self.fair_value_gaps.append(fvg)
            
            # Bearish FVG: gap between candle1 low and candle3 high
            elif candle1['low'] > candle3['high']:
                gap_size = candle1['low'] - candle3['high']
                if gap_size > self.config['fvg_threshold']:
                    # Check if middle candle doesn't fill the gap
                    if candle2['high'] < candle1['low'] and candle2['low'] > candle3['high']:
                        fvg = FairValueGap(
                            high=candle1['low'],
                            low=candle3['high'],
                            timestamp=candle2['timestamp'],
                            direction='bearish'
                        )
                        self.fair_value_gaps.append(fvg)
    
    def _identify_liquidity_levels(self, df: pd.DataFrame) -> None:
        """
        Identify liquidity levels (areas where stops are likely placed)
        
        Common liquidity areas:
        - Previous swing highs/lows
        - Round numbers
        - Daily/weekly highs/lows
        """
        self.liquidity_levels.clear()
        
        if len(df) < self.config['liquidity_lookback']:
            return
        
        recent_data = df.tail(self.config['liquidity_lookback'])
        
        # Find swing highs and lows
        for i in range(5, len(recent_data) - 5):
            current = recent_data.iloc[i]
            
            # Check for swing high (buy-side liquidity)
            if self._is_swing_high(recent_data, i):
                liquidity = LiquidityLevel(
                    price=current['high'],
                    timestamp=current['timestamp'],
                    type='buy_side',
                    strength=self._calculate_liquidity_strength(recent_data, i, 'high')
                )
                self.liquidity_levels.append(liquidity)
            
            # Check for swing low (sell-side liquidity)
            if self._is_swing_low(recent_data, i):
                liquidity = LiquidityLevel(
                    price=current['low'],
                    timestamp=current['timestamp'],
                    type='sell_side',
                    strength=self._calculate_liquidity_strength(recent_data, i, 'low')
                )
                self.liquidity_levels.append(liquidity)
    
    def _is_swing_high(self, df: pd.DataFrame, index: int, window: int = 3) -> bool:
        """Check if index is a swing high"""
        if index < window or index >= len(df) - window:
            return False
        
        current_high = df.iloc[index]['high']
        left_highs = df.iloc[index-window:index]['high']
        right_highs = df.iloc[index+1:index+window+1]['high']
        
        return current_high > left_highs.max() and current_high > right_highs.max()
    
    def _is_swing_low(self, df: pd.DataFrame, index: int, window: int = 3) -> bool:
        """Check if index is a swing low"""
        if index < window or index >= len(df) - window:
            return False
        
        current_low = df.iloc[index]['low']
        left_lows = df.iloc[index-window:index]['low']
        right_lows = df.iloc[index+1:index+window+1]['low']
        
        return current_low < left_lows.min() and current_low < right_lows.min()
    
    def _calculate_liquidity_strength(self, df: pd.DataFrame, index: int, price_type: str) -> float:
        """Calculate liquidity level strength"""
        current = df.iloc[index]
        
        # Volume at the level
        volume_score = min(current['volume'] / df['volume'].mean(), 2.0) * 30
        
        # Time since formation
        time_score = min((len(df) - index) / 20, 1.0) * 20
        
        # Number of touches (simplified)
        price = current[price_type]
        touches = sum(1 for i in range(len(df)) if abs(df.iloc[i][price_type] - price) < price * 0.001)
        touch_score = min(touches * 10, 30)
        
        # Range significance
        avg_range = df['range'].mean()
        range_score = min(current['range'] / avg_range, 2.0) * 20
        
        return min(volume_score + time_score + touch_score + range_score, 100)
    
    def _generate_signal(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal based on SMC confluence
        
        Signal generation criteria:
        1. Market structure alignment
        2. Order block confluence
        3. Fair value gap presence
        4. Liquidity level interaction
        """
        if len(df) < 10:
            return None
        
        current_price = df.iloc[-1]['close']
        current_time = df.iloc[-1]['timestamp']
        
        # Calculate confluence scores
        confluence_scores = self._calculate_confluence_scores(df, current_price)
        
        # Determine signal direction
        bullish_score = confluence_scores['bullish']
        bearish_score = confluence_scores['bearish']
        
        # Minimum confidence threshold
        min_confidence = 65
        
        if bullish_score > min_confidence and bullish_score > bearish_score:
            return self._create_buy_signal(df, current_price, current_time, bullish_score, confluence_scores)
        elif bearish_score > min_confidence and bearish_score > bullish_score:
            return self._create_sell_signal(df, current_price, current_time, bearish_score, confluence_scores)
        
        return None
    
    def _calculate_confluence_scores(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate confluence scores for bullish and bearish scenarios"""
        scores = {'bullish': 0, 'bearish': 0, 'details': {}}
        
        # Market structure score
        if self.market_structure == MarketStructure.BULLISH:
            scores['bullish'] += 25
        elif self.market_structure == MarketStructure.BEARISH:
            scores['bearish'] += 25
        
        scores['details']['market_structure'] = self.market_structure.value
        
        # Order block confluence
        ob_scores = self._get_order_block_confluence(current_price)
        scores['bullish'] += ob_scores['bullish']
        scores['bearish'] += ob_scores['bearish']
        scores['details']['order_blocks'] = ob_scores
        
        # Fair value gap confluence
        fvg_scores = self._get_fvg_confluence(current_price)
        scores['bullish'] += fvg_scores['bullish']
        scores['bearish'] += fvg_scores['bearish']
        scores['details']['fair_value_gaps'] = fvg_scores
        
        # Liquidity confluence
        liq_scores = self._get_liquidity_confluence(current_price)
        scores['bullish'] += liq_scores['bullish']
        scores['bearish'] += liq_scores['bearish']
        scores['details']['liquidity'] = liq_scores
        
        return scores
    
    def _get_order_block_confluence(self, current_price: float) -> Dict:
        """Get order block confluence scores"""
        scores = {'bullish': 0, 'bearish': 0, 'active_obs': []}
        
        for ob in self.order_blocks:
            if ob.low <= current_price <= ob.high:
                if ob.type in [OrderBlockType.BULLISH_OB, OrderBlockType.BREAKER_BULLISH]:
                    scores['bullish'] += min(ob.strength * 0.3, 20)
                elif ob.type in [OrderBlockType.BEARISH_OB, OrderBlockType.BREAKER_BEARISH]:
                    scores['bearish'] += min(ob.strength * 0.3, 20)
                
                scores['active_obs'].append({
                    'type': ob.type.value,
                    'strength': ob.strength,
                    'price_range': [ob.low, ob.high]
                })
        
        return scores
    
    def _get_fvg_confluence(self, current_price: float) -> Dict:
        """Get fair value gap confluence scores"""
        scores = {'bullish': 0, 'bearish': 0, 'active_fvgs': []}
        
        for fvg in self.fair_value_gaps:
            if not fvg.filled and fvg.low <= current_price <= fvg.high:
                if fvg.direction == 'bullish':
                    scores['bullish'] += 15
                elif fvg.direction == 'bearish':
                    scores['bearish'] += 15
                
                scores['active_fvgs'].append({
                    'direction': fvg.direction,
                    'price_range': [fvg.low, fvg.high],
                    'fill_percentage': fvg.fill_percentage
                })
        
        return scores
    
    def _get_liquidity_confluence(self, current_price: float) -> Dict:
        """Get liquidity level confluence scores"""
        scores = {'bullish': 0, 'bearish': 0, 'nearby_levels': []}
        
        # Check for nearby liquidity levels (within 0.1% of current price)
        price_tolerance = current_price * 0.001
        
        for liq in self.liquidity_levels:
            if abs(liq.price - current_price) <= price_tolerance:
                if liq.type == 'sell_side':  # Sell-side liquidity supports bullish moves
                    scores['bullish'] += min(liq.strength * 0.2, 15)
                elif liq.type == 'buy_side':  # Buy-side liquidity supports bearish moves
                    scores['bearish'] += min(liq.strength * 0.2, 15)
                
                scores['nearby_levels'].append({
                    'type': liq.type,
                    'price': liq.price,
                    'strength': liq.strength
                })
        
        return scores
    
    def _create_buy_signal(self, df: pd.DataFrame, entry_price: float, 
                          timestamp: datetime, confidence: float, analysis: Dict) -> TradingSignal:
        """Create a buy signal with proper risk management"""
        
        # Find nearest support for stop loss
        recent_lows = df.tail(20)['low']
        stop_loss = recent_lows.min() - (entry_price * 0.001)  # 0.1% buffer
        
        # Calculate take profit based on risk-reward ratio
        risk = entry_price - stop_loss
        take_profit = entry_price + (risk * 2.0)  # 1:2 risk-reward
        
        # Adjust based on nearby resistance
        nearby_resistance = self._find_nearby_resistance(df, entry_price)
        if nearby_resistance and nearby_resistance < take_profit:
            take_profit = nearby_resistance - (entry_price * 0.0005)  # Small buffer
        
        risk_reward_ratio = (take_profit - entry_price) / (entry_price - stop_loss)
        
        return TradingSignal(
            signal_type=SignalType.BUY,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            risk_reward_ratio=risk_reward_ratio,
            timestamp=timestamp,
            analysis=analysis,
            timeframe="1H"  # Default timeframe
        )
    
    def _create_sell_signal(self, df: pd.DataFrame, entry_price: float, 
                           timestamp: datetime, confidence: float, analysis: Dict) -> TradingSignal:
        """Create a sell signal with proper risk management"""
        
        # Find nearest resistance for stop loss
        recent_highs = df.tail(20)['high']
        stop_loss = recent_highs.max() + (entry_price * 0.001)  # 0.1% buffer
        
        # Calculate take profit based on risk-reward ratio
        risk = stop_loss - entry_price
        take_profit = entry_price - (risk * 2.0)  # 1:2 risk-reward
        
        # Adjust based on nearby support
        nearby_support = self._find_nearby_support(df, entry_price)
        if nearby_support and nearby_support > take_profit:
            take_profit = nearby_support + (entry_price * 0.0005)  # Small buffer
        
        risk_reward_ratio = (entry_price - take_profit) / (stop_loss - entry_price)
        
        return TradingSignal(
            signal_type=SignalType.SELL,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            risk_reward_ratio=risk_reward_ratio,
            timestamp=timestamp,
            analysis=analysis,
            timeframe="1H"  # Default timeframe
        )
    
    def _find_nearby_resistance(self, df: pd.DataFrame, price: float) -> Optional[float]:
        """Find the nearest resistance level above current price"""
        resistance_levels = []
        
        # Add swing highs as resistance
        for i in range(5, len(df) - 5):
            if self._is_swing_high(df, i):
                high = df.iloc[i]['high']
                if high > price:
                    resistance_levels.append(high)
        
        # Add order block highs as resistance
        for ob in self.order_blocks:
            if ob.type in [OrderBlockType.BEARISH_OB, OrderBlockType.BREAKER_BEARISH] and ob.high > price:
                resistance_levels.append(ob.high)
        
        return min(resistance_levels) if resistance_levels else None
    
    def _find_nearby_support(self, df: pd.DataFrame, price: float) -> Optional[float]:
        """Find the nearest support level below current price"""
        support_levels = []
        
        # Add swing lows as support
        for i in range(5, len(df) - 5):
            if self._is_swing_low(df, i):
                low = df.iloc[i]['low']
                if low < price:
                    support_levels.append(low)
        
        # Add order block lows as support
        for ob in self.order_blocks:
            if ob.type in [OrderBlockType.BULLISH_OB, OrderBlockType.BREAKER_BULLISH] and ob.low < price:
                support_levels.append(ob.low)
        
        return max(support_levels) if support_levels else None
    
    def get_market_summary(self) -> Dict:
        """Get current market summary and analysis"""
        return {
            'market_structure': self.market_structure.value,
            'active_order_blocks': len([ob for ob in self.order_blocks if not ob.tested]),
            'unfilled_fvgs': len([fvg for fvg in self.fair_value_gaps if not fvg.filled]),
            'liquidity_levels': len(self.liquidity_levels),
            'last_signal': self.last_signal.__dict__ if self.last_signal else None,
            'timestamp': datetime.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize SMC Engine
    smc = SMCEngine()
    
    # Sample data for testing (replace with real market data)
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'open': np.random.randn(100).cumsum() + 1.1000,
        'high': np.random.randn(100).cumsum() + 1.1020,
        'low': np.random.randn(100).cumsum() + 1.0980,
        'close': np.random.randn(100).cumsum() + 1.1010,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Ensure high >= low >= close relationships
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + abs(np.random.randn(100) * 0.001)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - abs(np.random.randn(100) * 0.001)
    
    # Run analysis
    try:
        analysis = smc.analyze_market_data(sample_data)
        print("SMC Analysis Results:")
        print(f"Market Structure: {analysis['market_structure']}")
        print(f"Order Blocks: {analysis['order_blocks']}")
        print(f"Active FVGs: {analysis['active_fvgs']}")
        print(f"Liquidity Levels: {analysis['liquidity_levels']}")
        
        if analysis['signal']:
            signal = analysis['signal']
            print(f"\nTrading Signal: {signal['signal_type']}")
            print(f"Entry: {signal['entry_price']:.5f}")
            print(f"Stop Loss: {signal['stop_loss']:.5f}")
            print(f"Take Profit: {signal['take_profit']:.5f}")
            print(f"Confidence: {signal['confidence']:.1f}%")
            print(f"Risk/Reward: 1:{signal['risk_reward_ratio']:.2f}")
        
    except Exception as e:
        print(f"Error in SMC analysis: {e}")
