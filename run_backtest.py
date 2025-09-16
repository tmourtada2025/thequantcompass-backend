#!/usr/bin/env python3
"""
The Quant Compass - Comprehensive Backtesting Script
====================================================

This script runs comprehensive backtests of our SMC trading model across multiple:
- Currency pairs (7 majors + 18 crosses)
- Timeframes (1H, 4H, 1D)
- Time periods (2+ years of data)
- Market conditions (trending, ranging, volatile)

The goal is to prove our SMC methodology works consistently and generates
profitable, risk-adjusted returns suitable for FTMO prop trading.
"""

import asyncio
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smc_engine import SMCEngine
from risk_manager import RiskManager
from market_data import PolygonDataProvider, BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    symbols: List[str]
    timeframes: List[str]
    start_date: str
    end_date: str
    initial_balance: float
    max_daily_risk: float
    max_total_risk: float
    commission_per_lot: float
    spread_pips: Dict[str, float]
    
@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    trades: List[Dict]
    equity_curve: List[Dict]
    monthly_returns: Dict[str, float]

class ComprehensiveBacktester:
    """
    Comprehensive backtesting system for The Quant Compass SMC model
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.smc_engine = SMCEngine()
        self.risk_manager = RiskManager(
            initial_balance=config.initial_balance,
            max_daily_risk=config.max_daily_risk,
            max_total_risk=config.max_total_risk
        )
        self.market_data = PolygonDataProvider()
        self.results: List[BacktestResult] = []
        
        # Create results directory
        self.results_dir = Path("backtest_results")
        self.results_dir.mkdir(exist_ok=True)
        
    async def run_comprehensive_backtest(self) -> Dict:
        """
        Run comprehensive backtests across all symbols and timeframes
        """
        logger.info("Starting comprehensive backtesting...")
        logger.info(f"Testing {len(self.config.symbols)} symbols across {len(self.config.timeframes)} timeframes")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        
        total_combinations = len(self.config.symbols) * len(self.config.timeframes)
        completed = 0
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                logger.info(f"Backtesting {symbol} on {timeframe} ({completed+1}/{total_combinations})")
                
                try:
                    result = await self.run_single_backtest(symbol, timeframe)
                    if result:
                        self.results.append(result)
                        logger.info(f"✅ {symbol} {timeframe}: {result.total_trades} trades, "
                                  f"{result.win_rate:.1f}% win rate, {result.total_pnl_pct:.2f}% return")
                    else:
                        logger.warning(f"❌ {symbol} {timeframe}: Backtest failed")
                        
                except Exception as e:
                    logger.error(f"❌ {symbol} {timeframe}: Error - {str(e)}")
                
                completed += 1
                
        # Generate comprehensive analysis
        analysis = self.analyze_results()
        
        # Save all results
        await self.save_results(analysis)
        
        logger.info("✅ Comprehensive backtesting completed!")
        return analysis
    
    async def run_single_backtest(self, symbol: str, timeframe: str) -> Optional[BacktestResult]:
        """
        Run backtest for a single symbol/timeframe combination
        """
        try:
            # Get historical data
            data = await self.market_data.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return None
            
            # Reset risk manager for each backtest
            self.risk_manager.reset_account(self.config.initial_balance)
            
            # Track trades and equity
            trades = []
            equity_curve = []
            current_balance = self.config.initial_balance
            
            # Track open positions
            open_positions = []
            
            # Process each candle
            for i in range(100, len(data)):  # Start after 100 candles for indicators
                current_time = data.index[i]
                current_data = data.iloc[:i+1]
                
                # Update equity curve
                equity_curve.append({
                    'timestamp': current_time.isoformat(),
                    'balance': current_balance,
                    'equity': current_balance,  # Simplified for now
                    'drawdown': max(0, (max([e['balance'] for e in equity_curve] + [current_balance]) - current_balance) / max([e['balance'] for e in equity_curve] + [current_balance]) * 100)
                })
                
                # Check for exit signals on open positions
                positions_to_close = []
                for pos in open_positions:
                    exit_signal = self.check_exit_conditions(pos, current_data, i)
                    if exit_signal:
                        positions_to_close.append(pos)
                
                # Close positions
                for pos in positions_to_close:
                    trade_result = self.close_position(pos, current_data.iloc[i], current_time)
                    trades.append(trade_result)
                    current_balance += trade_result['pnl']
                    open_positions.remove(pos)
                
                # Check for new entry signals
                if len(open_positions) == 0:  # Only one position at a time for now
                    signal = await self.smc_engine.analyze_market_structure(current_data)
                    
                    if signal and signal.get('signal') in ['BUY', 'SELL']:
                        # Calculate position size
                        risk_amount = self.risk_manager.calculate_position_size(
                            account_balance=current_balance,
                            stop_loss_pips=signal.get('stop_loss_pips', 50),
                            symbol=symbol
                        )
                        
                        if risk_amount > 0:
                            position = self.open_position(signal, current_data.iloc[i], current_time, risk_amount)
                            open_positions.append(position)
            
            # Close any remaining open positions
            for pos in open_positions:
                trade_result = self.close_position(pos, data.iloc[-1], data.index[-1])
                trades.append(trade_result)
                current_balance += trade_result['pnl']
            
            # Calculate performance metrics
            if len(trades) == 0:
                logger.warning(f"No trades generated for {symbol} {timeframe}")
                return None
                
            result = self.calculate_performance_metrics(
                symbol, timeframe, trades, equity_curve, current_balance
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in backtest for {symbol} {timeframe}: {str(e)}")
            return None
    
    def check_exit_conditions(self, position: Dict, data: pd.DataFrame, current_index: int) -> bool:
        """
        Check if position should be closed based on SMC exit rules
        """
        current_price = data.iloc[current_index]['close']
        entry_price = position['entry_price']
        signal_type = position['signal_type']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        # Basic stop loss / take profit
        if signal_type == 'BUY':
            if current_price <= stop_loss or current_price >= take_profit:
                return True
        else:  # SELL
            if current_price >= stop_loss or current_price <= take_profit:
                return True
                
        # Time-based exit (max 24 hours for intraday)
        entry_time = pd.to_datetime(position['entry_time'])
        current_time = data.index[current_index]
        if (current_time - entry_time).total_seconds() > 24 * 3600:
            return True
            
        return False
    
    def open_position(self, signal: Dict, candle: pd.Series, timestamp: pd.Timestamp, risk_amount: float) -> Dict:
        """
        Open a new position based on signal
        """
        entry_price = candle['close']
        signal_type = signal['signal']
        
        # Calculate stop loss and take profit
        stop_loss_pips = signal.get('stop_loss_pips', 50)
        take_profit_pips = signal.get('take_profit_pips', 100)
        
        pip_value = 0.0001  # Simplified for major pairs
        
        if signal_type == 'BUY':
            stop_loss = entry_price - (stop_loss_pips * pip_value)
            take_profit = entry_price + (take_profit_pips * pip_value)
        else:
            stop_loss = entry_price + (stop_loss_pips * pip_value)
            take_profit = entry_price - (take_profit_pips * pip_value)
        
        return {
            'entry_time': timestamp.isoformat(),
            'entry_price': entry_price,
            'signal_type': signal_type,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'lot_size': risk_amount / (stop_loss_pips * 10),  # Simplified calculation
            'signal_data': signal
        }
    
    def close_position(self, position: Dict, candle: pd.Series, timestamp: pd.Timestamp) -> Dict:
        """
        Close position and calculate P&L
        """
        exit_price = candle['close']
        entry_price = position['entry_price']
        signal_type = position['signal_type']
        lot_size = position['lot_size']
        
        # Calculate P&L
        if signal_type == 'BUY':
            pips = (exit_price - entry_price) / 0.0001
        else:
            pips = (entry_price - exit_price) / 0.0001
            
        pnl = pips * lot_size * 10  # Simplified P&L calculation
        
        # Subtract commission
        commission = self.config.commission_per_lot * lot_size
        pnl -= commission
        
        return {
            'entry_time': position['entry_time'],
            'exit_time': timestamp.isoformat(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'signal_type': signal_type,
            'lot_size': lot_size,
            'pips': pips,
            'pnl': pnl,
            'commission': commission,
            'duration_hours': (pd.to_datetime(timestamp) - pd.to_datetime(position['entry_time'])).total_seconds() / 3600
        }
    
    def calculate_performance_metrics(self, symbol: str, timeframe: str, trades: List[Dict], 
                                    equity_curve: List[Dict], final_balance: float) -> BacktestResult:
        """
        Calculate comprehensive performance metrics
        """
        if not trades:
            return None
            
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl = sum(t['pnl'] for t in trades)
        total_pnl_pct = (total_pnl / self.config.initial_balance) * 100
        
        # Win/Loss analysis
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown calculation
        equity_values = [e['balance'] for e in equity_curve]
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (running_max - equity_values) / running_max * 100
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        max_drawdown_pct = max_drawdown
        
        # Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = [t['pnl'] / self.config.initial_balance for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average trade duration
        durations = [t['duration_hours'] for t in trades]
        avg_trade_duration = np.mean(durations) if durations else 0
        
        # Monthly returns
        monthly_returns = self.calculate_monthly_returns(trades)
        
        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=abs(largest_loss),
            max_drawdown_pct=max_drawdown_pct,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns
        )
    
    def calculate_monthly_returns(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate monthly returns breakdown
        """
        monthly_pnl = {}
        
        for trade in trades:
            exit_date = pd.to_datetime(trade['exit_time'])
            month_key = exit_date.strftime('%Y-%m')
            
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += trade['pnl']
        
        # Convert to percentage returns
        monthly_returns = {}
        for month, pnl in monthly_pnl.items():
            monthly_returns[month] = (pnl / self.config.initial_balance) * 100
            
        return monthly_returns
    
    def analyze_results(self) -> Dict:
        """
        Analyze all backtest results and generate comprehensive report
        """
        if not self.results:
            return {"error": "No backtest results to analyze"}
        
        # Overall statistics
        total_trades = sum(r.total_trades for r in self.results)
        total_winning = sum(r.winning_trades for r in self.results)
        overall_win_rate = (total_winning / total_trades * 100) if total_trades > 0 else 0
        
        # Best and worst performers
        best_performer = max(self.results, key=lambda x: x.total_pnl_pct)
        worst_performer = min(self.results, key=lambda x: x.total_pnl_pct)
        
        # Average metrics
        avg_return = np.mean([r.total_pnl_pct for r in self.results])
        avg_win_rate = np.mean([r.win_rate for r in self.results])
        avg_profit_factor = np.mean([r.profit_factor for r in self.results if r.profit_factor > 0])
        avg_sharpe = np.mean([r.sharpe_ratio for r in self.results if r.sharpe_ratio != 0])
        
        # Risk metrics
        max_drawdown_overall = max([r.max_drawdown_pct for r in self.results])
        
        # Profitable vs unprofitable strategies
        profitable_strategies = len([r for r in self.results if r.total_pnl_pct > 0])
        total_strategies = len(self.results)
        strategy_success_rate = (profitable_strategies / total_strategies * 100) if total_strategies > 0 else 0
        
        analysis = {
            "summary": {
                "total_strategies_tested": total_strategies,
                "profitable_strategies": profitable_strategies,
                "strategy_success_rate": strategy_success_rate,
                "total_trades": total_trades,
                "overall_win_rate": overall_win_rate,
                "average_return": avg_return,
                "average_win_rate": avg_win_rate,
                "average_profit_factor": avg_profit_factor,
                "average_sharpe_ratio": avg_sharpe,
                "maximum_drawdown": max_drawdown_overall
            },
            "best_performer": {
                "symbol": best_performer.symbol,
                "timeframe": best_performer.timeframe,
                "return": best_performer.total_pnl_pct,
                "win_rate": best_performer.win_rate,
                "profit_factor": best_performer.profit_factor,
                "total_trades": best_performer.total_trades
            },
            "worst_performer": {
                "symbol": worst_performer.symbol,
                "timeframe": worst_performer.timeframe,
                "return": worst_performer.total_pnl_pct,
                "win_rate": worst_performer.win_rate,
                "profit_factor": worst_performer.profit_factor,
                "total_trades": worst_performer.total_trades
            },
            "by_symbol": self.analyze_by_symbol(),
            "by_timeframe": self.analyze_by_timeframe(),
            "risk_analysis": self.analyze_risk_metrics(),
            "recommendations": self.generate_recommendations()
        }
        
        return analysis
    
    def analyze_by_symbol(self) -> Dict:
        """Analyze performance by symbol"""
        symbol_performance = {}
        
        for symbol in self.config.symbols:
            symbol_results = [r for r in self.results if r.symbol == symbol]
            if symbol_results:
                avg_return = np.mean([r.total_pnl_pct for r in symbol_results])
                avg_win_rate = np.mean([r.win_rate for r in symbol_results])
                total_trades = sum([r.total_trades for r in symbol_results])
                
                symbol_performance[symbol] = {
                    "average_return": avg_return,
                    "average_win_rate": avg_win_rate,
                    "total_trades": total_trades,
                    "strategies_tested": len(symbol_results)
                }
        
        return symbol_performance
    
    def analyze_by_timeframe(self) -> Dict:
        """Analyze performance by timeframe"""
        timeframe_performance = {}
        
        for timeframe in self.config.timeframes:
            tf_results = [r for r in self.results if r.timeframe == timeframe]
            if tf_results:
                avg_return = np.mean([r.total_pnl_pct for r in tf_results])
                avg_win_rate = np.mean([r.win_rate for r in tf_results])
                total_trades = sum([r.total_trades for r in tf_results])
                
                timeframe_performance[timeframe] = {
                    "average_return": avg_return,
                    "average_win_rate": avg_win_rate,
                    "total_trades": total_trades,
                    "strategies_tested": len(tf_results)
                }
        
        return timeframe_performance
    
    def analyze_risk_metrics(self) -> Dict:
        """Analyze risk-related metrics"""
        returns = [r.total_pnl_pct for r in self.results]
        drawdowns = [r.max_drawdown_pct for r in self.results]
        
        return {
            "return_volatility": np.std(returns),
            "average_max_drawdown": np.mean(drawdowns),
            "worst_drawdown": max(drawdowns),
            "positive_return_probability": len([r for r in returns if r > 0]) / len(returns) * 100,
            "risk_adjusted_return": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        # Analyze results and generate recommendations
        profitable_results = [r for r in self.results if r.total_pnl_pct > 0]
        
        if len(profitable_results) / len(self.results) > 0.6:
            recommendations.append("✅ Strong overall performance - SMC methodology shows consistent profitability")
        else:
            recommendations.append("⚠️ Mixed results - consider parameter optimization or strategy refinement")
        
        # Best performing symbols
        symbol_performance = self.analyze_by_symbol()
        best_symbols = sorted(symbol_performance.items(), key=lambda x: x[1]['average_return'], reverse=True)[:3]
        recommendations.append(f"🎯 Focus on top performing symbols: {', '.join([s[0] for s in best_symbols])}")
        
        # Best performing timeframes
        tf_performance = self.analyze_by_timeframe()
        best_timeframes = sorted(tf_performance.items(), key=lambda x: x[1]['average_return'], reverse=True)[:2]
        recommendations.append(f"⏰ Optimal timeframes: {', '.join([tf[0] for tf in best_timeframes])}")
        
        # Risk management
        high_drawdown_results = [r for r in self.results if r.max_drawdown_pct > 10]
        if high_drawdown_results:
            recommendations.append("🛡️ Consider tighter stop losses - some strategies show high drawdown")
        
        return recommendations
    
    async def save_results(self, analysis: Dict):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_data = [asdict(result) for result in self.results]
        with open(self.results_dir / f"detailed_results_{timestamp}.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save analysis summary
        with open(self.results_dir / f"analysis_summary_{timestamp}.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate and save visualizations
        await self.create_visualizations(timestamp)
        
        # Generate text report
        self.generate_text_report(analysis, timestamp)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    async def create_visualizations(self, timestamp: str):
        """Create performance visualization charts"""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('The Quant Compass - SMC Backtesting Results', fontsize=16, fontweight='bold')
            
            # 1. Returns by Symbol
            symbol_returns = {}
            for result in self.results:
                if result.symbol not in symbol_returns:
                    symbol_returns[result.symbol] = []
                symbol_returns[result.symbol].append(result.total_pnl_pct)
            
            symbols = list(symbol_returns.keys())
            avg_returns = [np.mean(symbol_returns[symbol]) for symbol in symbols]
            
            axes[0, 0].bar(symbols, avg_returns, color='steelblue', alpha=0.7)
            axes[0, 0].set_title('Average Returns by Symbol')
            axes[0, 0].set_ylabel('Return (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Win Rate Distribution
            win_rates = [result.win_rate for result in self.results]
            axes[0, 1].hist(win_rates, bins=10, color='green', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Win Rate Distribution')
            axes[0, 1].set_xlabel('Win Rate (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(win_rates), color='red', linestyle='--', label=f'Average: {np.mean(win_rates):.1f}%')
            axes[0, 1].legend()
            
            # 3. Return vs Drawdown Scatter
            returns = [result.total_pnl_pct for result in self.results]
            drawdowns = [result.max_drawdown_pct for result in self.results]
            
            axes[1, 0].scatter(drawdowns, returns, alpha=0.6, color='purple')
            axes[1, 0].set_title('Return vs Maximum Drawdown')
            axes[1, 0].set_xlabel('Max Drawdown (%)')
            axes[1, 0].set_ylabel('Return (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Profit Factor Distribution
            profit_factors = [result.profit_factor for result in self.results if result.profit_factor > 0]
            axes[1, 1].hist(profit_factors, bins=10, color='orange', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Profit Factor Distribution')
            axes[1, 1].set_xlabel('Profit Factor')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(np.mean(profit_factors), color='red', linestyle='--', label=f'Average: {np.mean(profit_factors):.2f}')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f"performance_charts_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Performance charts saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def generate_text_report(self, analysis: Dict, timestamp: str):
        """Generate a comprehensive text report"""
        report = f"""
THE QUANT COMPASS - SMC BACKTESTING REPORT
==========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: {self.config.start_date} to {self.config.end_date}

EXECUTIVE SUMMARY
-----------------
Total Strategies Tested: {analysis['summary']['total_strategies_tested']}
Profitable Strategies: {analysis['summary']['profitable_strategies']} ({analysis['summary']['strategy_success_rate']:.1f}%)
Total Trades Executed: {analysis['summary']['total_trades']}
Overall Win Rate: {analysis['summary']['overall_win_rate']:.1f}%
Average Return: {analysis['summary']['average_return']:.2f}%
Average Profit Factor: {analysis['summary']['average_profit_factor']:.2f}
Maximum Drawdown: {analysis['summary']['maximum_drawdown']:.2f}%

BEST PERFORMING STRATEGY
-------------------------
Symbol: {analysis['best_performer']['symbol']}
Timeframe: {analysis['best_performer']['timeframe']}
Return: {analysis['best_performer']['return']:.2f}%
Win Rate: {analysis['best_performer']['win_rate']:.1f}%
Profit Factor: {analysis['best_performer']['profit_factor']:.2f}
Total Trades: {analysis['best_performer']['total_trades']}

PERFORMANCE BY SYMBOL
---------------------
"""
        
        for symbol, data in analysis['by_symbol'].items():
            report += f"{symbol}: {data['average_return']:.2f}% avg return, {data['average_win_rate']:.1f}% win rate\n"
        
        report += f"""
PERFORMANCE BY TIMEFRAME
------------------------
"""
        
        for timeframe, data in analysis['by_timeframe'].items():
            report += f"{timeframe}: {data['average_return']:.2f}% avg return, {data['average_win_rate']:.1f}% win rate\n"
        
        report += f"""
RISK ANALYSIS
-------------
Return Volatility: {analysis['risk_analysis']['return_volatility']:.2f}%
Average Max Drawdown: {analysis['risk_analysis']['average_max_drawdown']:.2f}%
Worst Drawdown: {analysis['risk_analysis']['worst_drawdown']:.2f}%
Positive Return Probability: {analysis['risk_analysis']['positive_return_probability']:.1f}%

RECOMMENDATIONS
---------------
"""
        
        for rec in analysis['recommendations']:
            report += f"• {rec}\n"
        
        report += f"""
DETAILED RESULTS
----------------
See detailed_results_{timestamp}.json for complete trade-by-trade data.
See performance_charts_{timestamp}.png for visual analysis.

DISCLAIMER
----------
Past performance is not indicative of future results. This backtesting was conducted
on historical data and may not reflect actual trading conditions. Always use proper
risk management when trading live accounts.
"""
        
        with open(self.results_dir / f"backtest_report_{timestamp}.txt", 'w') as f:
            f.write(report)
        
        logger.info("Text report generated successfully")

async def main():
    """
    Main function to run comprehensive backtesting
    """
    # Configuration for comprehensive backtesting
    config = BacktestConfig(
        symbols=[
            # Major Forex Pairs
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            # Cross Pairs (selection)
            'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'GBPAUD', 'AUDJPY',
            # Metals
            'XAUUSD', 'XAGUSD',
            # Indices (simplified symbols)
            'SPX500', 'NAS100', 'GER30',
            # Crypto (major)
            'BTCUSD', 'ETHUSD'
        ],
        timeframes=['1H', '4H', '1D'],
        start_date='2022-01-01',
        end_date='2024-12-31',
        initial_balance=100000.0,  # $100k FTMO account
        max_daily_risk=0.04,       # 4% daily risk (tighter than FTMO's 5%)
        max_total_risk=0.08,       # 8% total risk (tighter than FTMO's 10%)
        commission_per_lot=7.0,    # $7 per round trip
        spread_pips={
            'EURUSD': 1.0, 'GBPUSD': 1.5, 'USDJPY': 1.0, 'USDCHF': 1.5,
            'AUDUSD': 1.5, 'USDCAD': 1.5, 'NZDUSD': 2.0, 'EURJPY': 2.0,
            'GBPJPY': 2.5, 'EURGBP': 1.5, 'EURAUD': 2.0, 'GBPAUD': 3.0,
            'AUDJPY': 2.0, 'XAUUSD': 3.0, 'XAGUSD': 5.0, 'SPX500': 1.0,
            'NAS100': 2.0, 'GER30': 2.0, 'BTCUSD': 10.0, 'ETHUSD': 5.0
        }
    )
    
    # Initialize and run backtester
    backtester = ComprehensiveBacktester(config)
    
    try:
        logger.info("🚀 Starting The Quant Compass SMC Model Backtesting...")
        results = await backtester.run_comprehensive_backtest()
        
        logger.info("📊 Backtesting completed successfully!")
        logger.info(f"📈 Overall Results:")
        logger.info(f"   • Strategies Tested: {results['summary']['total_strategies_tested']}")
        logger.info(f"   • Profitable: {results['summary']['profitable_strategies']} ({results['summary']['strategy_success_rate']:.1f}%)")
        logger.info(f"   • Average Return: {results['summary']['average_return']:.2f}%")
        logger.info(f"   • Overall Win Rate: {results['summary']['overall_win_rate']:.1f}%")
        logger.info(f"   • Best Strategy: {results['best_performer']['symbol']} {results['best_performer']['timeframe']} ({results['best_performer']['return']:.2f}%)")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up environment
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Ensure we have the required API key
    if not os.getenv('POLYGON_API_KEY'):
        logger.error("POLYGON_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Run the backtesting
    asyncio.run(main())
