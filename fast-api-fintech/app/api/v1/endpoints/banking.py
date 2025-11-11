"""Banking endpoints for account and transaction management."""
from typing import List, Optional
from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, status, Query
from app.schemas.account import (
    Account, AccountCreate, AccountUpdate, AccountBalance, AccountStatus
)
from app.schemas.transaction import (
    Transaction, DepositCreate, WithdrawalCreate, TransferCreate,
    TransactionHistory, TransactionStatus
)
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/banking", tags=["Banking"])


# Account Management Endpoints

@router.post("/accounts", response_model=Account, status_code=status.HTTP_201_CREATED)
async def create_account(
    account_data: AccountCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new bank account.
    
    - **account_type**: CHECKING, SAVINGS, or INVESTMENT
    - **currency**: Currency code (default: USD)
    - **initial_balance**: Starting balance (must be >= 0)
    """
    import uuid
    from datetime import datetime
    
    account_number = f"ACC{uuid.uuid4().hex[:10].upper()}"
    
    return {
        "id": str(uuid.uuid4()),
        "user_id": current_user["user_id"],
        "account_number": account_number,
        "account_type": account_data.account_type,
        "balance": account_data.initial_balance,
        "currency": account_data.currency,
        "status": AccountStatus.ACTIVE,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@router.get("/accounts", response_model=List[Account])
async def get_accounts(
    current_user: dict = Depends(get_current_user),
    status_filter: Optional[AccountStatus] = None
):
    """
    Get all accounts for the current user.
    
    Optional filter by account status.
    """
    import uuid
    from datetime import datetime
    
    # Mock data - in production, fetch from database
    return [
        {
            "id": str(uuid.uuid4()),
            "user_id": current_user["user_id"],
            "account_number": "ACC1234567890",
            "account_type": "CHECKING",
            "balance": Decimal("5000.00"),
            "currency": "USD",
            "status": AccountStatus.ACTIVE,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]


@router.get("/accounts/{account_id}", response_model=Account)
async def get_account(
    account_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get account details by ID.
    """
    from datetime import datetime
    
    return {
        "id": account_id,
        "user_id": current_user["user_id"],
        "account_number": "ACC1234567890",
        "account_type": "CHECKING",
        "balance": Decimal("5000.00"),
        "currency": "USD",
        "status": AccountStatus.ACTIVE,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@router.get("/accounts/{account_id}/balance", response_model=AccountBalance)
async def get_account_balance(
    account_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get current balance for an account.
    """
    from datetime import datetime
    
    return {
        "account_id": account_id,
        "account_number": "ACC1234567890",
        "balance": Decimal("5000.00"),
        "currency": "USD",
        "as_of": datetime.utcnow()
    }


@router.patch("/accounts/{account_id}", response_model=Account)
async def update_account(
    account_id: str,
    account_update: AccountUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update account status (e.g., freeze or close account).
    """
    from datetime import datetime
    
    return {
        "id": account_id,
        "user_id": current_user["user_id"],
        "account_number": "ACC1234567890",
        "account_type": "CHECKING",
        "balance": Decimal("5000.00"),
        "currency": "USD",
        "status": account_update.status or AccountStatus.ACTIVE,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


# Transaction Endpoints

@router.post("/transactions/deposit", response_model=Transaction, status_code=status.HTTP_201_CREATED)
async def deposit(
    deposit_data: DepositCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Deposit money into an account.
    
    - **account_id**: Target account ID
    - **amount**: Amount to deposit (must be > 0)
    - **description**: Optional transaction description
    """
    import uuid
    from datetime import datetime
    
    # In production, verify account ownership and update balance
    new_balance = Decimal("5000.00") + deposit_data.amount
    
    return {
        "id": str(uuid.uuid4()),
        "account_id": deposit_data.account_id,
        "transaction_type": "DEPOSIT",
        "amount": deposit_data.amount,
        "balance_after": new_balance,
        "description": deposit_data.description,
        "status": TransactionStatus.COMPLETED,
        "created_at": datetime.utcnow()
    }


@router.post("/transactions/withdraw", response_model=Transaction, status_code=status.HTTP_201_CREATED)
async def withdraw(
    withdrawal_data: WithdrawalCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Withdraw money from an account.
    
    - **account_id**: Source account ID
    - **amount**: Amount to withdraw (must be > 0)
    - **description**: Optional transaction description
    """
    import uuid
    from datetime import datetime
    
    # In production, verify account ownership, check balance, and update
    current_balance = Decimal("5000.00")
    
    if withdrawal_data.amount > current_balance:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient funds"
        )
    
    new_balance = current_balance - withdrawal_data.amount
    
    return {
        "id": str(uuid.uuid4()),
        "account_id": withdrawal_data.account_id,
        "transaction_type": "WITHDRAWAL",
        "amount": withdrawal_data.amount,
        "balance_after": new_balance,
        "description": withdrawal_data.description,
        "status": TransactionStatus.COMPLETED,
        "created_at": datetime.utcnow()
    }


@router.post("/transactions/transfer", response_model=Transaction, status_code=status.HTTP_201_CREATED)
async def transfer(
    transfer_data: TransferCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Transfer money between accounts.
    
    - **from_account_id**: Source account ID
    - **to_account_id**: Destination account ID
    - **amount**: Amount to transfer (must be > 0)
    - **description**: Optional transaction description
    """
    import uuid
    from datetime import datetime
    
    # In production, verify both accounts, check balance, and perform transfer
    if transfer_data.from_account_id == transfer_data.to_account_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot transfer to the same account"
        )
    
    current_balance = Decimal("5000.00")
    
    if transfer_data.amount > current_balance:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient funds"
        )
    
    new_balance = current_balance - transfer_data.amount
    
    return {
        "id": str(uuid.uuid4()),
        "account_id": transfer_data.from_account_id,
        "transaction_type": "TRANSFER",
        "amount": transfer_data.amount,
        "balance_after": new_balance,
        "description": transfer_data.description or f"Transfer to {transfer_data.to_account_id}",
        "status": TransactionStatus.COMPLETED,
        "created_at": datetime.utcnow()
    }


@router.get("/transactions/history", response_model=TransactionHistory)
async def get_transaction_history(
    account_id: str = Query(..., description="Account ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of transactions to return"),
    offset: int = Query(0, ge=0, description="Number of transactions to skip"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get transaction history for an account.
    
    Supports pagination with limit and offset.
    """
    import uuid
    from datetime import datetime
    
    # Mock data - in production, fetch from database
    transactions = [
        {
            "id": str(uuid.uuid4()),
            "account_id": account_id,
            "transaction_type": "DEPOSIT",
            "amount": Decimal("1000.00"),
            "balance_after": Decimal("5000.00"),
            "description": "Initial deposit",
            "status": TransactionStatus.COMPLETED,
            "created_at": datetime.utcnow()
        }
    ]
    
    return {
        "transactions": transactions,
        "total": len(transactions),
        "page": offset // limit + 1,
        "page_size": limit
    }


@router.get("/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(
    transaction_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get transaction details by ID.
    """
    from datetime import datetime
    
    return {
        "id": transaction_id,
        "account_id": "acc_123",
        "transaction_type": "DEPOSIT",
        "amount": Decimal("1000.00"),
        "balance_after": Decimal("5000.00"),
        "description": "Sample transaction",
        "status": TransactionStatus.COMPLETED,
        "created_at": datetime.utcnow()
    }
