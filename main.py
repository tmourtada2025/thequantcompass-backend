"""
The Quant Compass - AI Trading Platform Backend
FastAPI application with Polygon.io integration, SMC analysis, and risk management
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

# Import our custom modules
from market_data import PolygonDataProvider, Timeframe, BacktestEngine
from smc_engine import SMCEngine, SMCSignal
from risk_manager import RiskManager, AccountPhase, RiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="The Quant Compass API",
    description="AI-powered trading platform with Smart Money Concepts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
polygon_provider: Optional[PolygonDataProvider] = None
smc_engine = SMCEngine()
risk_manager = RiskManager()

# Pydantic models for API requests/responses
class AccountUpdate(BaseModel):
    balance: float = Field(..., description="Current account balance")
    equity: float = Field(..., description="Current account equity")
    margin_used: float = Field(0.0, description="Margin currently used")
    daily_pnl: float = Field(0.0, description="Today's P&L")

class TradeRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    entry_price: float = Field(..., description="Planned entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    take_profit: float = Field(..., description="Take profit price")
    confidence: float = Field(..., ge=0, le=100, description="Signal confidence (0-100)")

class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field("H1", description="Timeframe (M1, M5, M15, M30, H1, H4, D1)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    strategy_params: Dict = Field(default_factory=dict, description="Strategy parameters")

class SignalResponse(BaseModel):
    symbol: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_reward_ratio: float
    analysis: Dict

class MarketDataResponse(BaseModel):
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global polygon_provider
    
    # Get Polygon API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        raise RuntimeError("Polygon API key not configured")
    
    # Initialize Polygon data provider
    polygon_provider = PolygonDataProvider(api_key)
    
    # Initialize risk manager with default account
    risk_manager.update_account_metrics(
        balance=100000.0,
        equity=100000.0,
        margin_used=0.0,
        daily_pnl=0.0
    )
    
    logger.info("The Quant Compass backend started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global polygon_provider
    if polygon_provider and polygon_provider.session:
        await polygon_provider.session.close()
    logger.info("The Quant Compass backend shutdown complete")

# Dependency to get Polygon provider
async def get_polygon_provider() -> PolygonDataProvider:
    if not polygon_provider:
        raise HTTPException(status_code=500, detail="Data provider not initialized")
    return polygon_provider

# Health check endpoint
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "The Quant Compass backend is live",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Market data endpoints
@app.get("/prices/{symbol}")
async def get_current_price(
    symbol: str,
    provider: PolygonDataProvider = Depends(get_polygon_provider)
):
    """Get current real-time price for a symbol"""
    try:
        quote = await provider.get_real_time_quote(symbol.upper())
        if not quote:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        return {
            "symbol": symbol.upper(),
            "quote": quote,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candles/{symbol}")
async def get_candles(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe (M1, M5, M15, M30, H1, H4, D1)"),
    limit: int = Query(100, ge=1, le=5000, description="Number of candles to retrieve"),
    provider: PolygonDataProvider = Depends(get_polygon_provider)
):
    """Get historical candle data for a symbol"""
    try:
        # Map timeframe string to enum
        timeframe_map = {
            "M1": Timeframe.M1, "M5": Timeframe.M5, "M15": Timeframe.M15,
            "M30": Timeframe.M30, "H1": Timeframe.H1, "H4": Timeframe.H4,
            "D1": Timeframe.D1, "W1": Timeframe.W1
        }
        
        if timeframe not in timeframe_map:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
        
        tf_enum = timeframe_map[timeframe]
        
        # Calculate date range based on timeframe and limit
        end_date = datetime.now()
        if tf_enum in [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30]:
            start_date = end_date - timedelta(days=7)  # 1 week for minute data
        elif tf_enum in [Timeframe.H1, Timeframe.H4]:
            start_date = end_date - timedelta(days=30)  # 1 month for hourly data
        else:
            start_date = end_date - timedelta(days=365)  # 1 year for daily data
        
        df = await provider.get_historical_data(
            symbol.upper(), tf_enum, start_date, end_date, limit
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convert DataFrame to list of dictionaries
        candles = []
        for _, row in df.tail(limit).iterrows():
            candles.append({
                "timestamp": row['timestamp'].isoformat(),
                "open": row['open'],
                "high": row['high'],
                "low": row['low'],
                "close": row['close'],
                "volume": row['volume']
            })
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "count": len(candles),
            "candles": candles
        }
        
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# SMC Analysis endpoints
@app.get("/analysis/{symbol}")
async def get_smc_analysis(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe for analysis"),
    provider: PolygonDataProvider = Depends(get_polygon_provider)
):
    """Get Smart Money Concepts analysis for a symbol"""
    try:
        # Get market data
        timeframe_map = {
            "M1": Timeframe.M1, "M5": Timeframe.M5, "M15": Timeframe.M15,
            "M30": Timeframe.M30, "H1": Timeframe.H1, "H4": Timeframe.H4,
            "D1": Timeframe.D1, "W1": Timeframe.W1
        }
        
        if timeframe not in timeframe_map:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
        
        tf_enum = timeframe_map[timeframe]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 days of data
        
        df = await provider.get_historical_data(
            symbol.upper(), tf_enum, start_date, end_date, 500
        )
        
        if df.empty or len(df) < 50:
            raise HTTPException(status_code=404, detail=f"Insufficient data for analysis of {symbol}")
        
        # Perform SMC analysis
        analysis = smc_engine.analyze_market_data(df)
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/{symbol}")
async def get_trading_signal(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe for signal generation"),
    provider: PolygonDataProvider = Depends(get_polygon_provider)
):
    """Generate trading signal for a symbol using SMC methodology"""
    try:
        # Get market data for analysis
        timeframe_map = {
            "M1": Timeframe.M1, "M5": Timeframe.M5, "M15": Timeframe.M15,
            "M30": Timeframe.M30, "H1": Timeframe.H1, "H4": Timeframe.H4,
            "D1": Timeframe.D1, "W1": Timeframe.W1
        }
        
        if timeframe not in timeframe_map:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
        
        tf_enum = timeframe_map[timeframe]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = await provider.get_historical_data(
            symbol.upper(), tf_enum, start_date, end_date, 500
        )
        
        if df.empty or len(df) < 50:
            raise HTTPException(status_code=404, detail=f"Insufficient data for signal generation of {symbol}")
        
        # Generate signal
        signal = smc_engine.generate_signal(df)
        
        if not signal:
            return {
                "symbol": symbol.upper(),
                "signal": None,
                "message": "No trading signal generated",
                "timestamp": datetime.now().isoformat()
            }
        
        # Assess trade risk
        trade_risk = risk_manager.assess_trade_risk(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            symbol=symbol.upper()
        )
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "signal": {
                "type": signal.signal_type.value,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "reasoning": signal.reasoning
            },
            "risk_assessment": {
                "approved": trade_risk.approved,
                "position_size": {
                    "lot_size": trade_risk.position_size.lot_size,
                    "risk_amount": trade_risk.position_size.risk_amount,
                    "risk_percentage": trade_risk.position_size.risk_percentage
                },
                "rejection_reason": trade_risk.rejection_reason
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk management endpoints
@app.post("/account/update")
async def update_account(account_data: AccountUpdate):
    """Update account metrics for risk management"""
    try:
        metrics = risk_manager.update_account_metrics(
            balance=account_data.balance,
            equity=account_data.equity,
            margin_used=account_data.margin_used,
            daily_pnl=account_data.daily_pnl
        )
        
        return {
            "status": "success",
            "account_metrics": {
                "current_balance": metrics.current_balance,
                "phase": metrics.phase.value,
                "risk_level": metrics.risk_level.value,
                "cumulative_drawdown": f"{metrics.cumulative_drawdown:.2f}%",
                "daily_pnl": metrics.daily_pnl
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating account: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/account/status")
async def get_account_status():
    """Get current account status and risk summary"""
    try:
        summary = risk_manager.get_risk_summary()
        emergency = risk_manager.check_emergency_stop()
        
        return {
            "account_summary": summary,
            "emergency_status": emergency,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting account status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trades/assess")
async def assess_trade_risk(trade_request: TradeRequest):
    """Assess risk for a potential trade"""
    try:
        trade_risk = risk_manager.assess_trade_risk(
            entry_price=trade_request.entry_price,
            stop_loss=trade_request.stop_loss,
            take_profit=trade_request.take_profit,
            confidence=trade_request.confidence,
            symbol=trade_request.symbol
        )
        
        return {
            "symbol": trade_request.symbol,
            "assessment": {
                "approved": trade_risk.approved,
                "risk_pips": trade_risk.risk_pips,
                "reward_pips": trade_risk.reward_pips,
                "risk_reward_ratio": trade_risk.risk_reward_ratio,
                "position_size": {
                    "lot_size": trade_risk.position_size.lot_size,
                    "risk_amount": trade_risk.position_size.risk_amount,
                    "risk_percentage": trade_risk.position_size.risk_percentage,
                    "reasoning": trade_risk.position_size.reasoning
                },
                "rejection_reason": trade_risk.rejection_reason
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error assessing trade risk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting endpoints
@app.post("/backtest/run")
async def run_backtest(
    backtest_request: BacktestRequest,
    background_tasks: BackgroundTasks,
    provider: PolygonDataProvider = Depends(get_polygon_provider)
):
    """Run a backtest for the specified parameters"""
    try:
        # Parse dates
        start_date = datetime.strptime(backtest_request.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(backtest_request.end_date, '%Y-%m-%d')
        
        # Map timeframe
        timeframe_map = {
            "M1": Timeframe.M1, "M5": Timeframe.M5, "M15": Timeframe.M15,
            "M30": Timeframe.M30, "H1": Timeframe.H1, "H4": Timeframe.H4,
            "D1": Timeframe.D1, "W1": Timeframe.W1
        }
        
        if backtest_request.timeframe not in timeframe_map:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe: {backtest_request.timeframe}")
        
        tf_enum = timeframe_map[backtest_request.timeframe]
        
        # Create backtest engine
        backtest_engine = BacktestEngine(provider)
        
        # Define strategy function
        def smc_strategy(df):
            return smc_engine.generate_signal(df)
        
        # Run backtest
        result = await backtest_engine.run_backtest(
            symbol=backtest_request.symbol.upper(),
            timeframe=tf_enum,
            start_date=start_date,
            end_date=end_date,
            strategy_func=smc_strategy,
            risk_manager=risk_manager
        )
        
        # Get performance summary
        summary = backtest_engine.get_performance_summary()
        
        return {
            "status": "completed",
            "backtest_id": f"{backtest_request.symbol}_{backtest_request.timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            "parameters": {
                "symbol": backtest_request.symbol.upper(),
                "timeframe": backtest_request.timeframe,
                "start_date": backtest_request.start_date,
                "end_date": backtest_request.end_date
            },
            "results": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/symbols")
async def get_supported_symbols():
    """Get list of supported trading symbols"""
    return {
        "forex_majors": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
        "forex_crosses": ["EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCHF", "AUDJPY", "CHFJPY"],
        "cryptocurrencies": ["BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "LINKUSD", "LTCUSD", "XRPUSD"],
        "commodities": ["XAUUSD", "XAGUSD", "XPTUSD", "USOIL", "UKOIL", "NGAS"],
        "indices": ["SPX", "DJI", "IXIC", "RUT", "VIX", "DAX", "FTSE", "N225"],
        "total_symbols": 35
    }

@app.get("/timeframes")
async def get_supported_timeframes():
    """Get list of supported timeframes"""
    return {
        "timeframes": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
        "descriptions": {
            "M1": "1 minute",
            "M5": "5 minutes", 
            "M15": "15 minutes",
            "M30": "30 minutes",
            "H1": "1 hour",
            "H4": "4 hours",
            "D1": "1 day",
            "W1": "1 week"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
