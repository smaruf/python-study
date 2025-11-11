"""Transaction schemas for banking operations."""
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    """Transaction type enumeration."""
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    TRANSFER = "TRANSFER"


class TransactionStatus(str, Enum):
    """Transaction status enumeration."""
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TransactionBase(BaseModel):
    """Base transaction schema."""
    amount: Decimal = Field(..., gt=0, description="Transaction amount must be positive")
    description: Optional[str] = None


class DepositCreate(TransactionBase):
    """Schema for deposit creation."""
    account_id: str


class WithdrawalCreate(TransactionBase):
    """Schema for withdrawal creation."""
    account_id: str


class TransferCreate(TransactionBase):
    """Schema for transfer creation."""
    from_account_id: str
    to_account_id: str


class TransactionInDB(TransactionBase):
    """Schema for transaction in database."""
    id: str
    account_id: str
    transaction_type: TransactionType
    balance_after: Decimal
    status: TransactionStatus
    created_at: datetime

    class Config:
        from_attributes = True


class Transaction(TransactionInDB):
    """Schema for transaction response."""
    pass


class TransactionHistory(BaseModel):
    """Schema for transaction history response."""
    transactions: list[Transaction]
    total: int
    page: int
    page_size: int
