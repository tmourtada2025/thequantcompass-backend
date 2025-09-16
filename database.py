"""
Database Models and Schemas
The Quant Compass - AI Trading Platform

SQLAlchemy models for storing trading data, user accounts, signals, and performance metrics.
"""

import os
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pydantic import BaseModel, Field
import uuid
import enum

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/quantcompass")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enums
class UserRole(enum.Enum):
    ADMIN = "admin"
    USER = "user"
    TRIAL = "trial"

class SubscriptionStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIAL = "trial"
    CANCELLED = "cancelled"

class SignalType(enum.Enum):
    BUY = "buy"
    SELL = "sell"

class SignalStatus(enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class TradeStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class AccountPhaseEnum(enum.Enum):
    GROWTH = "growth"
    RECOVERY = "recovery"
    DANGER = "danger"
    LOCKED = "locked"

# Database Models
class User(Base):
    """User account model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="user", uselist=False)
    trading_accounts = relationship("TradingAccount", back_populates="user")
    signals = relationship("Signal", back_populates="user")
    trades = relationship("Trade", back_populates="user")

class Subscription(Base):
    """User subscription model"""
    __tablename__ = "subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    plan_name = Column(String(100), nullable=False)  # Basic, Pro, Premium
    status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.TRIAL)
    price = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    billing_cycle = Column(String(20))  # monthly, yearly
    stripe_subscription_id = Column(String(255))
    stripe_customer_id = Column(String(255))
    trial_ends_at = Column(DateTime)
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    cancelled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="subscription")

class TradingAccount(Base):
    """Trading account model for risk management"""
    __tablename__ = "trading_accounts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    account_name = Column(String(100), nullable=False)
    broker = Column(String(100))  # FTMO, MyForexFunds, etc.
    account_type = Column(String(50))  # Demo, Live, Challenge
    starting_balance = Column(Float, nullable=False)
    current_balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    margin_used = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    cumulative_drawdown = Column(Float, default=0.0)
    max_daily_drawdown = Column(Float, default=5.0)
    max_cumulative_drawdown = Column(Float, default=10.0)
    account_phase = Column(SQLEnum(AccountPhaseEnum), default=AccountPhaseEnum.GROWTH)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="trading_accounts")
    trades = relationship("Trade", back_populates="trading_account")

class Signal(Base):
    """Trading signal model"""
    __tablename__ = "signals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)  # Null for public signals
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(SQLEnum(SignalType), nullable=False)
    timeframe = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)  # 0-100
    risk_reward_ratio = Column(Float, nullable=False)
    reasoning = Column(Text)
    smc_analysis = Column(JSONB)  # Store SMC analysis data
    status = Column(SQLEnum(SignalStatus), default=SignalStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    activated_at = Column(DateTime)
    closed_at = Column(DateTime)
    exit_price = Column(Float)
    exit_reason = Column(String(50))  # stop_loss, take_profit, manual, expired
    pnl_pips = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="signals")
    trades = relationship("Trade", back_populates="signal")

class Trade(Base):
    """Individual trade execution model"""
    __tablename__ = "trades"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    trading_account_id = Column(UUID(as_uuid=True), ForeignKey("trading_accounts.id"), nullable=False)
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(SQLEnum(SignalType), nullable=False)
    lot_size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    exit_price = Column(Float)
    status = Column(SQLEnum(TradeStatus), default=TradeStatus.OPEN)
    risk_amount = Column(Float, nullable=False)
    risk_percentage = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=datetime.utcnow, index=True)
    closed_at = Column(DateTime)
    exit_reason = Column(String(50))
    mt4_ticket = Column(String(50))  # MT4/MT5 ticket number if applicable
    
    # Relationships
    user = relationship("User", back_populates="trades")
    trading_account = relationship("TradingAccount", back_populates="trades")
    signal = relationship("Signal", back_populates="trades")

class BacktestResult(Base):
    """Backtest results model"""
    __tablename__ = "backtest_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    name = Column(String(255), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    strategy_params = Column(JSONB)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    equity_curve = Column(JSONB)  # Store equity curve data
    trade_details = Column(JSONB)  # Store individual trade data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")

class MarketData(Base):
    """Market data cache model"""
    __tablename__ = "market_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite index for efficient queries
    __table_args__ = (
        {'extend_existing': True}
    )

class SystemSettings(Base):
    """System configuration model"""
    __tablename__ = "system_settings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(Text)
    category = Column(String(50), default="general")
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic schemas for API
class UserCreate(BaseModel):
    email: str = Field(..., description="User email address")
    username: str = Field(..., description="Unique username")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: Optional[str] = Field(None, description="Full name")

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class TradingAccountCreate(BaseModel):
    account_name: str = Field(..., description="Account display name")
    broker: Optional[str] = Field(None, description="Broker name")
    account_type: str = Field("Demo", description="Account type")
    starting_balance: float = Field(..., gt=0, description="Starting balance")
    max_daily_drawdown: float = Field(5.0, description="Max daily drawdown %")
    max_cumulative_drawdown: float = Field(10.0, description="Max cumulative drawdown %")

class TradingAccountResponse(BaseModel):
    id: str
    account_name: str
    broker: Optional[str]
    account_type: str
    starting_balance: float
    current_balance: float
    equity: float
    cumulative_drawdown: float
    account_phase: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class SignalCreate(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    signal_type: str = Field(..., description="buy or sell")
    timeframe: str = Field(..., description="Timeframe")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    take_profit: float = Field(..., description="Take profit price")
    confidence: float = Field(..., ge=0, le=100, description="Confidence 0-100")
    reasoning: Optional[str] = Field(None, description="Signal reasoning")

class SignalResponse(BaseModel):
    id: str
    symbol: str
    signal_type: str
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_reward_ratio: float
    reasoning: Optional[str]
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class TradeCreate(BaseModel):
    trading_account_id: str = Field(..., description="Trading account ID")
    signal_id: Optional[str] = Field(None, description="Related signal ID")
    symbol: str = Field(..., description="Trading symbol")
    direction: str = Field(..., description="buy or sell")
    lot_size: float = Field(..., gt=0, description="Position size in lots")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    take_profit: float = Field(..., description="Take profit price")
    risk_amount: float = Field(..., description="Risk amount in account currency")

class TradeResponse(BaseModel):
    id: str
    symbol: str
    direction: str
    lot_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: Optional[float]
    status: str
    risk_amount: float
    pnl: float
    opened_at: datetime
    closed_at: Optional[datetime]
    
    class Config:
        from_attributes = True

# Database utility functions
def get_db():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """Drop all database tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)

# Initialize database
if __name__ == "__main__":
    print("Creating database tables...")
    create_tables()
    print("Database tables created successfully!")
