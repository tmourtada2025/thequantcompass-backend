"""
Risk Management System
The Quant Compass - AI Trading Platform

FTMO-compliant risk management system with dynamic position sizing,
drawdown protection, and adaptive risk controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AccountPhase(Enum):
    """Account trading phases"""
    GROWTH = "growth"          # Account above starting balance
    RECOVERY = "recovery"      # Account below starting balance
    DANGER = "danger"          # Near maximum drawdown
    LOCKED = "locked"          # Trading disabled due to limits


class RiskLevel(Enum):
    """Risk level classifications"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class AccountMetrics:
    """Current account state metrics"""
    current_balance: float
    starting_balance: float
    daily_pnl: float
    max_daily_drawdown: float
    cumulative_drawdown: float
    max_cumulative_drawdown: float
    equity: float
    margin_used: float
    free_margin: float
    phase: AccountPhase
    risk_level: RiskLevel


@dataclass
class PositionSize:
    """Position sizing calculation result"""
    lot_size: float
    risk_amount: float
    risk_percentage: float
    position_value: float
    margin_required: float
    max_loss: float
    reasoning: str


@dataclass
class TradeRisk:
    """Individual trade risk assessment"""
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pips: float
    reward_pips: float
    risk_reward_ratio: float
    position_size: PositionSize
    approved: bool
    rejection_reason: Optional[str] = None


class RiskManager:
    """
    FTMO-Compliant Risk Management System
    
    Features:
    - Dynamic position sizing based on account phase
    - Daily and cumulative drawdown protection
    - Adaptive risk levels
    - Trade approval/rejection system
    - Real-time risk monitoring
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.account_metrics: Optional[AccountMetrics] = None
        self.daily_trades: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.last_update = datetime.now()
        
    def _default_config(self) -> Dict:
        """Default FTMO-compliant risk configuration"""
        return {
            # FTMO Rules (Conservative Settings)
            'starting_balance': 100000.0,
            'max_daily_drawdown_pct': 4.0,      # 4% instead of 5% for safety
            'max_cumulative_drawdown_pct': 8.0,  # 8% instead of 10% for safety
            
            # Position Sizing Rules
            'growth_phase': {
                'base_risk_pct': 1.0,           # 1% risk per trade in growth
                'max_risk_pct': 2.0,            # Maximum 2% per trade
                'max_daily_risk_pct': 3.0,      # Maximum 3% daily risk
                'max_positions': 5              # Maximum concurrent positions
            },
            'recovery_phase': {
                'base_risk_pct': 0.5,           # 0.5% risk per trade in recovery
                'max_risk_pct': 1.0,            # Maximum 1% per trade
                'max_daily_risk_pct': 2.0,      # Maximum 2% daily risk
                'max_positions': 3              # Maximum concurrent positions
            },
            'danger_phase': {
                'base_risk_pct': 0.25,          # 0.25% risk per trade near limits
                'max_risk_pct': 0.5,            # Maximum 0.5% per trade
                'max_daily_risk_pct': 1.0,      # Maximum 1% daily risk
                'max_positions': 2              # Maximum concurrent positions
            },
            
            # Risk Adjustment Factors
            'confidence_multiplier': {
                'high': 1.2,        # Increase size for high confidence
                'medium': 1.0,      # Normal size for medium confidence
                'low': 0.7          # Reduce size for low confidence
            },
            
            # Market Condition Adjustments
            'volatility_adjustment': True,
            'correlation_limit': 0.7,           # Max correlation between positions
            'news_risk_reduction': 0.5,         # Reduce size during high-impact news
            
            # Safety Limits
            'min_risk_reward': 1.5,             # Minimum 1:1.5 risk/reward
            'max_spread_pips': 3.0,             # Maximum spread for entry
            'min_stop_distance_pips': 10.0,     # Minimum stop loss distance
            'max_stop_distance_pips': 100.0     # Maximum stop loss distance
        }
    
    def update_account_metrics(self, balance: float, equity: float, 
                             margin_used: float = 0.0, daily_pnl: float = 0.0) -> AccountMetrics:
        """Update current account metrics and determine trading phase"""
        
        starting_balance = self.config['starting_balance']
        
        # Calculate drawdowns
        balance_drawdown = max(0, (starting_balance - balance) / starting_balance * 100)
        equity_drawdown = max(0, (starting_balance - equity) / starting_balance * 100)
        cumulative_drawdown = max(balance_drawdown, equity_drawdown)
        
        # Calculate daily drawdown (simplified - would need daily starting balance in production)
        daily_drawdown_pct = abs(daily_pnl) / starting_balance * 100 if daily_pnl < 0 else 0
        
        # Determine account phase
        phase = self._determine_account_phase(balance, cumulative_drawdown, daily_drawdown_pct)
        
        # Determine risk level
        risk_level = self._determine_risk_level(phase, cumulative_drawdown)
        
        self.account_metrics = AccountMetrics(
            current_balance=balance,
            starting_balance=starting_balance,
            daily_pnl=daily_pnl,
            max_daily_drawdown=self.config['max_daily_drawdown_pct'],
            cumulative_drawdown=cumulative_drawdown,
            max_cumulative_drawdown=self.config['max_cumulative_drawdown_pct'],
            equity=equity,
            margin_used=margin_used,
            free_margin=equity - margin_used,
            phase=phase,
            risk_level=risk_level
        )
        
        logger.info(f"Account updated: {phase.value} phase, {cumulative_drawdown:.2f}% drawdown")
        return self.account_metrics
    
    def _determine_account_phase(self, balance: float, cumulative_drawdown: float, 
                               daily_drawdown: float) -> AccountPhase:
        """Determine current account trading phase"""
        
        # Check if near danger limits
        if (cumulative_drawdown >= self.config['max_cumulative_drawdown_pct'] * 0.8 or
            daily_drawdown >= self.config['max_daily_drawdown_pct'] * 0.8):
            return AccountPhase.DANGER
        
        # Check if limits exceeded (should stop trading)
        if (cumulative_drawdown >= self.config['max_cumulative_drawdown_pct'] or
            daily_drawdown >= self.config['max_daily_drawdown_pct']):
            return AccountPhase.LOCKED
        
        # Check if in recovery (below starting balance)
        if balance < self.config['starting_balance']:
            return AccountPhase.RECOVERY
        
        # Growth phase (above starting balance)
        return AccountPhase.GROWTH
    
    def _determine_risk_level(self, phase: AccountPhase, cumulative_drawdown: float) -> RiskLevel:
        """Determine appropriate risk level based on account state"""
        
        if phase == AccountPhase.LOCKED or phase == AccountPhase.DANGER:
            return RiskLevel.CONSERVATIVE
        elif phase == AccountPhase.RECOVERY:
            return RiskLevel.CONSERVATIVE if cumulative_drawdown > 5.0 else RiskLevel.MODERATE
        else:  # GROWTH phase
            return RiskLevel.MODERATE if cumulative_drawdown < 2.0 else RiskLevel.CONSERVATIVE
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              confidence: float, symbol: str = "EURUSD") -> PositionSize:
        """
        Calculate optimal position size based on current account state
        
        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss price
            confidence: Signal confidence (0-100)
            symbol: Trading symbol for pip value calculation
            
        Returns:
            PositionSize object with calculated lot size and risk metrics
        """
        
        if not self.account_metrics:
            raise ValueError("Account metrics must be updated before calculating position size")
        
        if self.account_metrics.phase == AccountPhase.LOCKED:
            return PositionSize(
                lot_size=0.0,
                risk_amount=0.0,
                risk_percentage=0.0,
                position_value=0.0,
                margin_required=0.0,
                max_loss=0.0,
                reasoning="Trading locked due to drawdown limits"
            )
        
        # Get phase-specific risk parameters
        phase_config = self._get_phase_config()
        
        # Calculate base risk amount
        base_risk_pct = phase_config['base_risk_pct']
        
        # Adjust for confidence
        confidence_factor = self._get_confidence_factor(confidence)
        adjusted_risk_pct = base_risk_pct * confidence_factor
        
        # Apply maximum risk limits
        max_risk_pct = phase_config['max_risk_pct']
        final_risk_pct = min(adjusted_risk_pct, max_risk_pct)
        
        # Calculate risk amount in account currency
        risk_amount = self.account_metrics.current_balance * (final_risk_pct / 100)
        
        # Calculate pip value and position size
        pip_risk = abs(entry_price - stop_loss)
        pip_value = self._calculate_pip_value(symbol, entry_price)
        
        # Calculate lot size
        if pip_risk > 0 and pip_value > 0:
            lot_size = risk_amount / (pip_risk * pip_value * 100000)  # Standard lot calculation
            lot_size = round(lot_size, 2)  # Round to 2 decimal places
        else:
            lot_size = 0.0
        
        # Calculate position metrics
        position_value = lot_size * 100000 * entry_price  # For major pairs
        margin_required = position_value * 0.02  # Assuming 50:1 leverage (2% margin)
        max_loss = lot_size * pip_risk * pip_value * 100000
        
        # Validate position size
        reasoning = self._validate_position_size(lot_size, margin_required, max_loss, phase_config)
        
        return PositionSize(
            lot_size=lot_size,
            risk_amount=risk_amount,
            risk_percentage=final_risk_pct,
            position_value=position_value,
            margin_required=margin_required,
            max_loss=max_loss,
            reasoning=reasoning
        )
    
    def _get_phase_config(self) -> Dict:
        """Get risk configuration for current account phase"""
        phase_map = {
            AccountPhase.GROWTH: 'growth_phase',
            AccountPhase.RECOVERY: 'recovery_phase',
            AccountPhase.DANGER: 'danger_phase',
            AccountPhase.LOCKED: 'danger_phase'  # Use danger config for locked
        }
        
        phase_key = phase_map[self.account_metrics.phase]
        return self.config[phase_key]
    
    def _get_confidence_factor(self, confidence: float) -> float:
        """Get position size multiplier based on signal confidence"""
        if confidence >= 80:
            return self.config['confidence_multiplier']['high']
        elif confidence >= 60:
            return self.config['confidence_multiplier']['medium']
        else:
            return self.config['confidence_multiplier']['low']
    
    def _calculate_pip_value(self, symbol: str, price: float) -> float:
        """Calculate pip value for position sizing (simplified)"""
        # Simplified pip value calculation for major pairs
        # In production, this would use real-time exchange rates
        
        if symbol.endswith('USD'):
            return 10.0  # $10 per pip for standard lot
        elif symbol.startswith('USD'):
            return 10.0 / price  # Adjusted for USD base
        else:
            return 10.0  # Default for cross pairs (simplified)
    
    def _validate_position_size(self, lot_size: float, margin_required: float, 
                              max_loss: float, phase_config: Dict) -> str:
        """Validate calculated position size against risk limits"""
        
        reasons = []
        
        # Check minimum lot size
        if lot_size < 0.01:
            reasons.append("Position size below minimum (0.01 lots)")
        
        # Check margin requirements
        if margin_required > self.account_metrics.free_margin * 0.8:
            reasons.append("Insufficient margin (using >80% of free margin)")
        
        # Check daily risk limits
        daily_risk_used = self._calculate_daily_risk_used()
        if daily_risk_used + (max_loss / self.account_metrics.current_balance * 100) > phase_config['max_daily_risk_pct']:
            reasons.append(f"Exceeds daily risk limit ({phase_config['max_daily_risk_pct']}%)")
        
        # Check position count limits
        open_positions = len([t for t in self.daily_trades if t.get('status') == 'open'])
        if open_positions >= phase_config['max_positions']:
            reasons.append(f"Maximum positions reached ({phase_config['max_positions']})")
        
        if reasons:
            return "; ".join(reasons)
        else:
            return f"Approved for {self.account_metrics.phase.value} phase trading"
    
    def _calculate_daily_risk_used(self) -> float:
        """Calculate percentage of daily risk already used"""
        today = datetime.now().date()
        today_trades = [t for t in self.daily_trades if t.get('date', datetime.now().date()) == today]
        
        total_risk = sum(t.get('risk_amount', 0) for t in today_trades)
        return (total_risk / self.account_metrics.current_balance) * 100
    
    def assess_trade_risk(self, entry_price: float, stop_loss: float, 
                         take_profit: float, confidence: float, 
                         symbol: str = "EURUSD") -> TradeRisk:
        """
        Comprehensive trade risk assessment
        
        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: Signal confidence (0-100)
            symbol: Trading symbol
            
        Returns:
            TradeRisk object with approval/rejection decision
        """
        
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss, confidence, symbol)
        
        # Calculate risk/reward metrics
        risk_pips = abs(entry_price - stop_loss)
        reward_pips = abs(take_profit - entry_price)
        risk_reward_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
        
        # Initial approval based on position size
        approved = position_size.lot_size > 0
        rejection_reason = None
        
        # Additional risk checks
        if approved:
            # Check minimum risk/reward ratio
            if risk_reward_ratio < self.config['min_risk_reward']:
                approved = False
                rejection_reason = f"Risk/reward ratio too low ({risk_reward_ratio:.2f} < {self.config['min_risk_reward']})"
            
            # Check stop loss distance
            elif risk_pips < self.config['min_stop_distance_pips']:
                approved = False
                rejection_reason = f"Stop loss too close ({risk_pips:.1f} < {self.config['min_stop_distance_pips']} pips)"
            
            elif risk_pips > self.config['max_stop_distance_pips']:
                approved = False
                rejection_reason = f"Stop loss too far ({risk_pips:.1f} > {self.config['max_stop_distance_pips']} pips)"
            
            # Check account phase restrictions
            elif self.account_metrics.phase == AccountPhase.LOCKED:
                approved = False
                rejection_reason = "Trading locked due to drawdown limits"
        
        else:
            rejection_reason = position_size.reasoning
        
        trade_risk = TradeRisk(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pips=risk_pips,
            reward_pips=reward_pips,
            risk_reward_ratio=risk_reward_ratio,
            position_size=position_size,
            approved=approved,
            rejection_reason=rejection_reason
        )
        
        # Log the decision
        if approved:
            logger.info(f"Trade APPROVED: {position_size.lot_size} lots, {risk_reward_ratio:.2f} R:R, {confidence:.0f}% confidence")
        else:
            logger.warning(f"Trade REJECTED: {rejection_reason}")
        
        return trade_risk
    
    def record_trade(self, trade_data: Dict) -> None:
        """Record a trade for risk tracking"""
        trade_record = {
            'timestamp': datetime.now(),
            'date': datetime.now().date(),
            'symbol': trade_data.get('symbol'),
            'direction': trade_data.get('direction'),
            'lot_size': trade_data.get('lot_size'),
            'entry_price': trade_data.get('entry_price'),
            'stop_loss': trade_data.get('stop_loss'),
            'take_profit': trade_data.get('take_profit'),
            'risk_amount': trade_data.get('risk_amount'),
            'status': trade_data.get('status', 'open'),
            'pnl': trade_data.get('pnl', 0.0)
        }
        
        self.daily_trades.append(trade_record)
        self.trade_history.append(trade_record)
        
        # Clean up old daily trades (keep only today's)
        today = datetime.now().date()
        self.daily_trades = [t for t in self.daily_trades if t['date'] == today]
    
    def get_risk_summary(self) -> Dict:
        """Get current risk management summary"""
        if not self.account_metrics:
            return {'error': 'Account metrics not initialized'}
        
        daily_risk_used = self._calculate_daily_risk_used()
        phase_config = self._get_phase_config()
        
        return {
            'account_phase': self.account_metrics.phase.value,
            'risk_level': self.account_metrics.risk_level.value,
            'current_balance': self.account_metrics.current_balance,
            'cumulative_drawdown': f"{self.account_metrics.cumulative_drawdown:.2f}%",
            'daily_pnl': self.account_metrics.daily_pnl,
            'daily_risk_used': f"{daily_risk_used:.2f}%",
            'daily_risk_available': f"{phase_config['max_daily_risk_pct'] - daily_risk_used:.2f}%",
            'max_position_size': f"{phase_config['max_risk_pct']}%",
            'open_positions': len([t for t in self.daily_trades if t.get('status') == 'open']),
            'max_positions': phase_config['max_positions'],
            'trading_allowed': self.account_metrics.phase != AccountPhase.LOCKED,
            'last_update': self.last_update.isoformat()
        }
    
    def check_emergency_stop(self) -> Dict:
        """Check if emergency stop conditions are met"""
        if not self.account_metrics:
            return {'emergency_stop': False, 'reason': 'No account data'}
        
        emergency_conditions = []
        
        # Check drawdown limits
        if self.account_metrics.cumulative_drawdown >= self.config['max_cumulative_drawdown_pct']:
            emergency_conditions.append(f"Cumulative drawdown limit exceeded ({self.account_metrics.cumulative_drawdown:.2f}%)")
        
        # Check daily drawdown
        daily_drawdown = abs(self.account_metrics.daily_pnl) / self.account_metrics.starting_balance * 100
        if self.account_metrics.daily_pnl < 0 and daily_drawdown >= self.config['max_daily_drawdown_pct']:
            emergency_conditions.append(f"Daily drawdown limit exceeded ({daily_drawdown:.2f}%)")
        
        # Check margin level
        if self.account_metrics.margin_used > 0:
            margin_level = (self.account_metrics.equity / self.account_metrics.margin_used) * 100
            if margin_level < 150:  # 150% margin level threshold
                emergency_conditions.append(f"Low margin level ({margin_level:.0f}%)")
        
        return {
            'emergency_stop': len(emergency_conditions) > 0,
            'conditions': emergency_conditions,
            'action_required': 'Close all positions immediately' if emergency_conditions else None
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager()
    
    # Update account metrics (example)
    risk_manager.update_account_metrics(
        balance=98500.0,    # $1,500 drawdown
        equity=98200.0,     # $300 floating loss
        margin_used=5000.0,
        daily_pnl=-800.0    # $800 daily loss
    )
    
    # Test trade risk assessment
    trade_risk = risk_manager.assess_trade_risk(
        entry_price=1.1000,
        stop_loss=1.0950,   # 50 pip stop
        take_profit=1.1100, # 100 pip target (1:2 R:R)
        confidence=75.0,
        symbol="EURUSD"
    )
    
    print("Risk Assessment Results:")
    print(f"Approved: {trade_risk.approved}")
    print(f"Position Size: {trade_risk.position_size.lot_size} lots")
    print(f"Risk Amount: ${trade_risk.position_size.risk_amount:.2f}")
    print(f"Risk/Reward: 1:{trade_risk.risk_reward_ratio:.2f}")
    
    if not trade_risk.approved:
        print(f"Rejection Reason: {trade_risk.rejection_reason}")
    
    # Get risk summary
    summary = risk_manager.get_risk_summary()
    print(f"\nRisk Summary:")
    print(f"Account Phase: {summary['account_phase']}")
    print(f"Cumulative Drawdown: {summary['cumulative_drawdown']}")
    print(f"Daily Risk Used: {summary['daily_risk_used']}")
    
    # Check emergency conditions
    emergency = risk_manager.check_emergency_stop()
    if emergency['emergency_stop']:
        print(f"\n⚠️  EMERGENCY STOP: {emergency['conditions']}")
