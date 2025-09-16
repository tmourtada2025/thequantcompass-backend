"""
AI Analyst Module
The Quant Compass - AI Trading Platform

OpenAI integration for generating market analysis, trade reasoning, and research reports.
This module powers the "Seeking Alpha-style" research capabilities.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import openai
from dataclasses import dataclass
import pandas as pd

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class MarketAnalysis:
    """Market analysis result structure"""
    symbol: str
    timeframe: str
    analysis_type: str
    summary: str
    detailed_analysis: str
    key_levels: Dict[str, float]
    sentiment: str  # bullish, bearish, neutral
    confidence: float  # 0-100
    trade_ideas: List[Dict[str, Any]]
    risk_factors: List[str]
    timestamp: datetime

@dataclass
class TradingReport:
    """Comprehensive trading report structure"""
    title: str
    executive_summary: str
    market_overview: str
    technical_analysis: str
    fundamental_factors: str
    trade_recommendations: List[Dict[str, Any]]
    risk_assessment: str
    conclusion: str
    charts_needed: List[str]
    timestamp: datetime

class AIAnalyst:
    """AI-powered market analyst using OpenAI GPT models"""
    
    def __init__(self):
        self.model = "gpt-4"
        self.max_tokens = 4000
        self.temperature = 0.3  # Lower temperature for more consistent analysis
        
    async def analyze_market_structure(self, symbol: str, timeframe: str, 
                                     price_data: pd.DataFrame, 
                                     smc_analysis: Dict[str, Any]) -> MarketAnalysis:
        """
        Generate comprehensive market analysis using SMC methodology
        """
        try:
            # Prepare market data context
            latest_price = price_data.iloc[-1]
            price_change = ((latest_price['close'] - price_data.iloc[-20]['close']) / 
                          price_data.iloc[-20]['close'] * 100)
            
            # Create analysis prompt
            prompt = self._create_market_analysis_prompt(
                symbol, timeframe, latest_price, price_change, smc_analysis
            )
            
            # Get AI analysis
            response = await self._call_openai(prompt)
            
            # Parse and structure the response
            analysis = self._parse_market_analysis(response, symbol, timeframe)
            
            return analysis
            
        except Exception as e:
            print(f"Error in market analysis: {e}")
            return self._create_fallback_analysis(symbol, timeframe)
    
    async def generate_trading_report(self, symbols: List[str], 
                                    market_data: Dict[str, pd.DataFrame],
                                    performance_metrics: Dict[str, Any]) -> TradingReport:
        """
        Generate a comprehensive "Seeking Alpha-style" trading report
        """
        try:
            # Create comprehensive report prompt
            prompt = self._create_report_prompt(symbols, market_data, performance_metrics)
            
            # Get AI-generated report
            response = await self._call_openai(prompt, max_tokens=6000)
            
            # Parse and structure the report
            report = self._parse_trading_report(response)
            
            return report
            
        except Exception as e:
            print(f"Error generating trading report: {e}")
            return self._create_fallback_report()
    
    async def explain_trade_signal(self, signal_data: Dict[str, Any], 
                                 market_context: Dict[str, Any]) -> str:
        """
        Generate detailed explanation for a trading signal
        """
        try:
            prompt = f"""
            As an expert Smart Money Concepts trader, explain this trading signal in detail:
            
            SIGNAL DETAILS:
            Symbol: {signal_data.get('symbol')}
            Direction: {signal_data.get('signal', {}).get('type', 'N/A').upper()}
            Entry: {signal_data.get('signal', {}).get('entry_price', 'N/A')}
            Stop Loss: {signal_data.get('signal', {}).get('stop_loss', 'N/A')}
            Take Profit: {signal_data.get('signal', {}).get('take_profit', 'N/A')}
            Confidence: {signal_data.get('signal', {}).get('confidence', 'N/A')}%
            Risk/Reward: {signal_data.get('signal', {}).get('risk_reward_ratio', 'N/A')}
            
            MARKET CONTEXT:
            {json.dumps(market_context, indent=2)}
            
            Please provide:
            1. Why this signal was generated (SMC perspective)
            2. Key market structure elements identified
            3. Risk factors to consider
            4. What to watch for during the trade
            5. Exit strategy considerations
            
            Keep the explanation professional but accessible, as if writing for serious traders.
            """
            
            response = await self._call_openai(prompt, max_tokens=1500)
            return response.strip()
            
        except Exception as e:
            print(f"Error explaining trade signal: {e}")
            return "Signal analysis temporarily unavailable. Please refer to the technical indicators."
    
    async def generate_market_outlook(self, timeframe: str = "weekly") -> str:
        """
        Generate a market outlook report for multiple asset classes
        """
        try:
            prompt = f"""
            As a professional market analyst specializing in Smart Money Concepts, 
            provide a {timeframe} market outlook covering:
            
            1. FOREX MAJORS (EUR/USD, GBP/USD, USD/JPY, etc.)
            2. PRECIOUS METALS (Gold, Silver)
            3. ENERGY (Crude Oil, Natural Gas)
            4. MAJOR INDICES (S&P 500, NASDAQ, DAX)
            5. CRYPTOCURRENCIES (Bitcoin, Ethereum)
            
            For each asset class, discuss:
            - Current market structure and trend
            - Key support/resistance levels
            - Smart money activity indicators
            - Potential trading opportunities
            - Risk factors to monitor
            
            Format as a professional market report suitable for institutional traders.
            Current date: {datetime.now().strftime('%Y-%m-%d')}
            """
            
            response = await self._call_openai(prompt, max_tokens=4000)
            return response.strip()
            
        except Exception as e:
            print(f"Error generating market outlook: {e}")
            return "Market outlook temporarily unavailable."
    
    async def analyze_performance_insights(self, backtest_results: List[Dict[str, Any]]) -> str:
        """
        Generate insights from backtesting performance data
        """
        try:
            # Summarize performance data
            total_trades = sum(r.get('results', {}).get('total_trades', 0) for r in backtest_results)
            avg_win_rate = sum(float(r.get('results', {}).get('win_rate', '0').replace('%', '')) 
                             for r in backtest_results) / len(backtest_results) if backtest_results else 0
            
            prompt = f"""
            As a quantitative analyst, analyze these backtesting results and provide insights:
            
            PERFORMANCE SUMMARY:
            Total Backtests: {len(backtest_results)}
            Total Trades: {total_trades}
            Average Win Rate: {avg_win_rate:.1f}%
            
            DETAILED RESULTS:
            {json.dumps(backtest_results, indent=2)}
            
            Please provide:
            1. Key performance insights
            2. Strengths of the strategy
            3. Areas for improvement
            4. Risk assessment
            5. Recommendations for optimization
            6. Market conditions where strategy performs best/worst
            
            Format as a professional performance analysis report.
            """
            
            response = await self._call_openai(prompt, max_tokens=2000)
            return response.strip()
            
        except Exception as e:
            print(f"Error analyzing performance: {e}")
            return "Performance analysis temporarily unavailable."
    
    def _create_market_analysis_prompt(self, symbol: str, timeframe: str, 
                                     latest_price: pd.Series, price_change: float,
                                     smc_analysis: Dict[str, Any]) -> str:
        """Create a detailed prompt for market analysis"""
        return f"""
        As an expert Smart Money Concepts (SMC) trader, analyze {symbol} on {timeframe} timeframe:
        
        CURRENT MARKET DATA:
        Price: {latest_price['close']:.5f}
        Change: {price_change:+.2f}%
        High: {latest_price['high']:.5f}
        Low: {latest_price['low']:.5f}
        Volume: {latest_price.get('volume', 'N/A')}
        
        SMC ANALYSIS:
        {json.dumps(smc_analysis, indent=2)}
        
        Provide a comprehensive analysis including:
        1. Market structure assessment (bullish/bearish/neutral)
        2. Key support and resistance levels
        3. Order block identification
        4. Liquidity zones
        5. Potential trade setups
        6. Risk factors
        7. Confidence level (0-100)
        
        Format as JSON with clear sections for each analysis component.
        """
    
    def _create_report_prompt(self, symbols: List[str], 
                            market_data: Dict[str, pd.DataFrame],
                            performance_metrics: Dict[str, Any]) -> str:
        """Create a comprehensive trading report prompt"""
        return f"""
        Create a professional trading research report titled "The Quant Compass Weekly Market Analysis" 
        in the style of Seeking Alpha or institutional research.
        
        SYMBOLS TO ANALYZE: {', '.join(symbols)}
        
        PERFORMANCE CONTEXT:
        {json.dumps(performance_metrics, indent=2)}
        
        Structure the report with:
        1. Executive Summary
        2. Market Overview
        3. Technical Analysis (SMC methodology)
        4. Trade Recommendations
        5. Risk Assessment
        6. Conclusion
        
        Write in a professional, analytical tone suitable for serious traders and institutions.
        Include specific price levels, risk/reward ratios, and actionable insights.
        """
    
    async def _call_openai(self, prompt: str, max_tokens: int = None) -> str:
        """Make an async call to OpenAI API"""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional trading analyst specializing in Smart Money Concepts methodology."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    def _parse_market_analysis(self, response: str, symbol: str, timeframe: str) -> MarketAnalysis:
        """Parse AI response into structured market analysis"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                data = json.loads(response)
                return MarketAnalysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    analysis_type="smc_analysis",
                    summary=data.get('summary', response[:200]),
                    detailed_analysis=data.get('detailed_analysis', response),
                    key_levels=data.get('key_levels', {}),
                    sentiment=data.get('sentiment', 'neutral'),
                    confidence=data.get('confidence', 75.0),
                    trade_ideas=data.get('trade_ideas', []),
                    risk_factors=data.get('risk_factors', []),
                    timestamp=datetime.now()
                )
            else:
                # Parse as text
                return MarketAnalysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    analysis_type="smc_analysis",
                    summary=response[:200] + "..." if len(response) > 200 else response,
                    detailed_analysis=response,
                    key_levels={},
                    sentiment="neutral",
                    confidence=75.0,
                    trade_ideas=[],
                    risk_factors=[],
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            print(f"Error parsing market analysis: {e}")
            return self._create_fallback_analysis(symbol, timeframe)
    
    def _parse_trading_report(self, response: str) -> TradingReport:
        """Parse AI response into structured trading report"""
        try:
            # Split response into sections
            sections = response.split('\n\n')
            
            return TradingReport(
                title="The Quant Compass Market Analysis",
                executive_summary=sections[0] if len(sections) > 0 else "Analysis in progress...",
                market_overview=sections[1] if len(sections) > 1 else "",
                technical_analysis=sections[2] if len(sections) > 2 else "",
                fundamental_factors=sections[3] if len(sections) > 3 else "",
                trade_recommendations=[],
                risk_assessment=sections[-2] if len(sections) > 4 else "",
                conclusion=sections[-1] if len(sections) > 0 else "",
                charts_needed=["price_chart", "volume_profile", "market_structure"],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error parsing trading report: {e}")
            return self._create_fallback_report()
    
    def _create_fallback_analysis(self, symbol: str, timeframe: str) -> MarketAnalysis:
        """Create a fallback analysis when AI fails"""
        return MarketAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            analysis_type="fallback",
            summary="Market analysis temporarily unavailable",
            detailed_analysis="Please check back later for detailed analysis",
            key_levels={},
            sentiment="neutral",
            confidence=0.0,
            trade_ideas=[],
            risk_factors=["Analysis service unavailable"],
            timestamp=datetime.now()
        )
    
    def _create_fallback_report(self) -> TradingReport:
        """Create a fallback report when AI fails"""
        return TradingReport(
            title="Market Analysis - Service Unavailable",
            executive_summary="Market analysis service is temporarily unavailable",
            market_overview="",
            technical_analysis="",
            fundamental_factors="",
            trade_recommendations=[],
            risk_assessment="",
            conclusion="Please try again later",
            charts_needed=[],
            timestamp=datetime.now()
        )

# Utility functions for integration
async def generate_signal_explanation(signal_data: Dict[str, Any]) -> str:
    """Generate explanation for a trading signal"""
    analyst = AIAnalyst()
    return await analyst.explain_trade_signal(signal_data, {})

async def generate_weekly_report(symbols: List[str]) -> TradingReport:
    """Generate weekly market report"""
    analyst = AIAnalyst()
    return await analyst.generate_trading_report(symbols, {}, {})

async def analyze_strategy_performance(backtest_results: List[Dict[str, Any]]) -> str:
    """Analyze backtesting performance"""
    analyst = AIAnalyst()
    return await analyst.analyze_performance_insights(backtest_results)

# Example usage
if __name__ == "__main__":
    async def test_ai_analyst():
        analyst = AIAnalyst()
        
        # Test signal explanation
        signal_data = {
            "symbol": "EURUSD",
            "signal": {
                "type": "buy",
                "entry_price": 1.0850,
                "stop_loss": 1.0800,
                "take_profit": 1.0950,
                "confidence": 85,
                "risk_reward_ratio": 2.0
            }
        }
        
        explanation = await analyst.explain_trade_signal(signal_data, {})
        print("Signal Explanation:")
        print(explanation)
        print("\n" + "="*50 + "\n")
        
        # Test market outlook
        outlook = await analyst.generate_market_outlook("weekly")
        print("Market Outlook:")
        print(outlook)
    
    # Run test
    # asyncio.run(test_ai_analyst())
