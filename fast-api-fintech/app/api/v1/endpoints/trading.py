"""Trading endpoints for stock exchange operations."""
from typing import List, Optional
from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, status, Query
from app.schemas.stock import Stock, StockList
from app.schemas.order import (
    Order, OrderCreate, OrderUpdate, OrderList, Trade, Portfolio,
    OrderStatus, OrderSide
)
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/trading", tags=["Trading"])


# Market Data Endpoints

@router.get("/market/stocks", response_model=StockList)
async def get_market_stocks(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """
    Get list of available stocks with current market data.
    """
    import uuid
    from datetime import datetime
    
    # Mock data - in production, fetch from database or external market data API
    stocks = [
        {
            "id": str(uuid.uuid4()),
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "current_price": Decimal("175.50"),
            "previous_close": Decimal("173.25"),
            "market_cap": Decimal("2750000000000"),
            "volume": 45000000,
            "updated_at": datetime.utcnow(),
            "change": Decimal("2.25"),
            "change_percent": Decimal("1.30")
        },
        {
            "id": str(uuid.uuid4()),
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "current_price": Decimal("140.25"),
            "previous_close": Decimal("139.50"),
            "market_cap": Decimal("1750000000000"),
            "volume": 28000000,
            "updated_at": datetime.utcnow(),
            "change": Decimal("0.75"),
            "change_percent": Decimal("0.54")
        },
        {
            "id": str(uuid.uuid4()),
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "current_price": Decimal("385.75"),
            "previous_close": Decimal("383.00"),
            "market_cap": Decimal("2850000000000"),
            "volume": 32000000,
            "updated_at": datetime.utcnow(),
            "change": Decimal("2.75"),
            "change_percent": Decimal("0.72")
        }
    ]
    
    return {
        "stocks": stocks,
        "total": len(stocks)
    }


@router.get("/market/stocks/{symbol}", response_model=Stock)
async def get_stock(
    symbol: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information for a specific stock.
    """
    import uuid
    from datetime import datetime
    
    # Mock data - in production, fetch from database or market data API
    if symbol.upper() not in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock {symbol} not found"
        )
    
    return {
        "id": str(uuid.uuid4()),
        "symbol": symbol.upper(),
        "name": "Apple Inc." if symbol.upper() == "AAPL" else "Stock Company",
        "current_price": Decimal("175.50"),
        "previous_close": Decimal("173.25"),
        "market_cap": Decimal("2750000000000"),
        "volume": 45000000,
        "updated_at": datetime.utcnow(),
        "change": Decimal("2.25"),
        "change_percent": Decimal("1.30")
    }


# Order Management Endpoints

@router.post("/orders", response_model=Order, status_code=status.HTTP_201_CREATED)
async def create_order(
    order_data: OrderCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Place a new stock order.
    
    - **symbol**: Stock symbol (e.g., AAPL)
    - **order_type**: MARKET, LIMIT, STOP, or STOP_LIMIT
    - **side**: BUY or SELL
    - **quantity**: Number of shares (must be > 0)
    - **price**: Price per share (required for LIMIT orders)
    """
    import uuid
    from datetime import datetime
    
    # Validate price is provided for LIMIT orders
    if order_data.order_type in ["LIMIT", "STOP_LIMIT"] and not order_data.price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Price is required for {order_data.order_type} orders"
        )
    
    # In production, validate stock exists, check account balance, and place order
    return {
        "id": str(uuid.uuid4()),
        "user_id": current_user["user_id"],
        "symbol": order_data.symbol.upper(),
        "order_type": order_data.order_type,
        "side": order_data.side,
        "quantity": order_data.quantity,
        "price": order_data.price,
        "status": OrderStatus.PENDING,
        "filled_quantity": 0,
        "average_price": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@router.get("/orders", response_model=OrderList)
async def get_orders(
    status_filter: Optional[OrderStatus] = None,
    symbol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all orders for the current user.
    
    Optional filters by status and symbol.
    """
    import uuid
    from datetime import datetime
    
    # Mock data - in production, fetch from database with filters
    orders = [
        {
            "id": str(uuid.uuid4()),
            "user_id": current_user["user_id"],
            "symbol": "AAPL",
            "order_type": "LIMIT",
            "side": "BUY",
            "quantity": 10,
            "price": Decimal("175.00"),
            "status": OrderStatus.FILLED,
            "filled_quantity": 10,
            "average_price": Decimal("175.50"),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    ]
    
    return {
        "orders": orders,
        "total": len(orders)
    }


@router.get("/orders/{order_id}", response_model=Order)
async def get_order(
    order_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get order details by ID.
    """
    from datetime import datetime
    
    return {
        "id": order_id,
        "user_id": current_user["user_id"],
        "symbol": "AAPL",
        "order_type": "LIMIT",
        "side": "BUY",
        "quantity": 10,
        "price": Decimal("175.00"),
        "status": OrderStatus.FILLED,
        "filled_quantity": 10,
        "average_price": Decimal("175.50"),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@router.patch("/orders/{order_id}", response_model=Order)
async def update_order(
    order_id: str,
    order_update: OrderUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update an existing order (modify quantity or price).
    
    Only PENDING orders can be modified.
    """
    from datetime import datetime
    
    # In production, verify order ownership and status
    return {
        "id": order_id,
        "user_id": current_user["user_id"],
        "symbol": "AAPL",
        "order_type": "LIMIT",
        "side": "BUY",
        "quantity": order_update.quantity or 10,
        "price": order_update.price or Decimal("175.00"),
        "status": OrderStatus.PENDING,
        "filled_quantity": 0,
        "average_price": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@router.delete("/orders/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_order(
    order_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Cancel an existing order.
    
    Only PENDING or PARTIALLY_FILLED orders can be cancelled.
    """
    # In production, verify order ownership, status, and cancel
    return None


# Portfolio and Trading History

@router.get("/portfolio", response_model=Portfolio)
async def get_portfolio(
    current_user: dict = Depends(get_current_user)
):
    """
    Get current portfolio with all holdings.
    """
    # Mock data - in production, calculate from filled orders
    holdings = [
        {
            "symbol": "AAPL",
            "quantity": 10,
            "average_cost": Decimal("175.00"),
            "current_price": Decimal("175.50"),
            "total_value": Decimal("1755.00"),
            "unrealized_pnl": Decimal("5.00")
        },
        {
            "symbol": "GOOGL",
            "quantity": 5,
            "average_cost": Decimal("140.00"),
            "current_price": Decimal("140.25"),
            "total_value": Decimal("701.25"),
            "unrealized_pnl": Decimal("1.25")
        }
    ]
    
    total_value = sum(h["total_value"] for h in holdings)
    
    return {
        "holdings": holdings,
        "total_value": total_value,
        "cash_balance": Decimal("10000.00")
    }


@router.get("/trades/history", response_model=List[Trade])
async def get_trade_history(
    symbol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """
    Get trade execution history.
    
    Optional filter by symbol.
    """
    import uuid
    from datetime import datetime
    
    # Mock data - in production, fetch completed trades from database
    trades = [
        {
            "id": str(uuid.uuid4()),
            "order_id": str(uuid.uuid4()),
            "symbol": "AAPL",
            "side": OrderSide.BUY,
            "quantity": 10,
            "price": Decimal("175.50"),
            "total_value": Decimal("1755.00"),
            "executed_at": datetime.utcnow()
        }
    ]
    
    return trades
