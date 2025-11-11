"""Account schemas for banking operations."""
from typing import Optional
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field


class AccountType(str, Enum):
    """Account type enumeration."""
    CHECKING = "CHECKING"
    SAVINGS = "SAVINGS"
    INVESTMENT = "INVESTMENT"


class AccountStatus(str, Enum):
    """Account status enumeration."""
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    CLOSED = "CLOSED"


class AccountBase(BaseModel):
    """Base account schema."""
    account_type: AccountType
    currency: str = "USD"


class AccountCreate(AccountBase):
    """Schema for account creation."""
    initial_balance: Decimal = Field(default=Decimal("0.00"), ge=0)


class AccountUpdate(BaseModel):
    """Schema for account update."""
    status: Optional[AccountStatus] = None


class AccountInDB(AccountBase):
    """Schema for account in database."""
    id: str
    user_id: str
    account_number: str
    balance: Decimal
    status: AccountStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Account(AccountInDB):
    """Schema for account response."""
    pass


class AccountBalance(BaseModel):
    """Schema for account balance response."""
    account_id: str
    account_number: str
    balance: Decimal
    currency: str
    as_of: datetime
