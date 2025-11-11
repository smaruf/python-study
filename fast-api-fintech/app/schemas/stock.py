"""Stock schemas for trading operations."""
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field


class StockBase(BaseModel):
    """Base stock schema."""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol (e.g., AAPL)")
    name: str


class StockCreate(StockBase):
    """Schema for stock creation."""
    current_price: Decimal = Field(..., gt=0)
    previous_close: Decimal = Field(..., gt=0)
    market_cap: Decimal = Field(..., gt=0)
    volume: int = Field(..., ge=0)


class StockUpdate(BaseModel):
    """Schema for stock update."""
    current_price: Decimal = Field(..., gt=0)
    volume: int = Field(..., ge=0)


class StockInDB(StockBase):
    """Schema for stock in database."""
    id: str
    current_price: Decimal
    previous_close: Decimal
    market_cap: Decimal
    volume: int
    updated_at: datetime

    class Config:
        from_attributes = True


class Stock(StockInDB):
    """Schema for stock response."""
    change: Decimal = Field(default=Decimal("0.00"))
    change_percent: Decimal = Field(default=Decimal("0.00"))

    @property
    def calculated_change(self) -> Decimal:
        """Calculate price change."""
        return self.current_price - self.previous_close

    @property
    def calculated_change_percent(self) -> Decimal:
        """Calculate percentage change."""
        if self.previous_close > 0:
            return ((self.current_price - self.previous_close) / self.previous_close) * 100
        return Decimal("0.00")


class StockList(BaseModel):
    """Schema for stock list response."""
    stocks: list[Stock]
    total: int
