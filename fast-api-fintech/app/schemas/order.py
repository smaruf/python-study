"""Order schemas for trading operations."""
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderBase(BaseModel):
    """Base order schema."""
    symbol: str = Field(..., min_length=1, max_length=10)
    order_type: OrderType
    side: OrderSide
    quantity: int = Field(..., gt=0)
    price: Optional[Decimal] = Field(None, gt=0)


class OrderCreate(OrderBase):
    """Schema for order creation."""
    pass


class OrderUpdate(BaseModel):
    """Schema for order update."""
    quantity: Optional[int] = Field(None, gt=0)
    price: Optional[Decimal] = Field(None, gt=0)


class OrderInDB(OrderBase):
    """Schema for order in database."""
    id: str
    user_id: str
    status: OrderStatus
    filled_quantity: int = 0
    average_price: Optional[Decimal] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Order(OrderInDB):
    """Schema for order response."""
    pass


class OrderList(BaseModel):
    """Schema for order list response."""
    orders: list[Order]
    total: int


class Trade(BaseModel):
    """Schema for trade response."""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    total_value: Decimal
    executed_at: datetime

    class Config:
        from_attributes = True


class Portfolio(BaseModel):
    """Schema for portfolio response."""
    holdings: list[dict]
    total_value: Decimal
    cash_balance: Decimal
