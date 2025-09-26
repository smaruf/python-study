"""Orders API endpoints for Phase 1."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

from core.models.order import Order, OrderSide, OrderType
from core.services.oms_service import OMSService

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])
oms_service = OMSService()


class OrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: int
    price: Optional[float] = None


class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: int
    price: Optional[float]
    status: str
    created_at: datetime


@router.post("/", response_model=dict)
async def submit_order(order_request: OrderRequest):
    """Submit a new order."""
    try:
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=order_request.symbol,
            side=OrderSide(order_request.side),
            order_type=OrderType(order_request.order_type),
            quantity=order_request.quantity,
            price=order_request.price,
            created_at=datetime.utcnow()
        )
        
        result = await oms_service.submit_order(order)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """Get order by ID."""
    order = await oms_service.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return OrderResponse(
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side.value,
        order_type=order.order_type.value,
        quantity=order.quantity,
        price=order.price,
        status=order.status.value,
        created_at=order.created_at
    )


@router.get("/", response_model=List[OrderResponse])
async def get_orders():
    """Get all orders."""
    orders = await oms_service.get_orders()
    return [
        OrderResponse(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=order.quantity,
            price=order.price,
            status=order.status.value,
            created_at=order.created_at
        )
        for order in orders
    ]


@router.delete("/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order."""
    result = await oms_service.cancel_order(order_id)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result
