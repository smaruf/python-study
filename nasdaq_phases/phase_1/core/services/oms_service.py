"""Order Management System service - Phase 1 implementation."""
import asyncio
from typing import Dict, List, Optional
from core.models.order import Order, OrderStatus
from core.repositories.order_repository import OrderRepository


class OMSService:
    """Basic Order Management System for Phase 1."""
    
    def __init__(self):
        self.order_repository = OrderRepository()
        self._orders: Dict[str, Order] = {}
    
    async def submit_order(self, order: Order) -> Dict[str, str]:
        """Submit a new order."""
        try:
            # Basic validation
            if not order.symbol:
                return {"status": "error", "message": "Symbol is required"}
            
            if order.quantity <= 0:
                return {"status": "error", "message": "Quantity must be positive"}
            
            # Store order
            self._orders[order.order_id] = order
            await self.order_repository.save(order)
            
            # Set status to pending
            order.status = OrderStatus.PENDING
            
            return {
                "status": "success", 
                "order_id": order.order_id,
                "message": "Order submitted successfully"
            }
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    async def get_orders(self) -> List[Order]:
        """Get all orders."""
        return list(self._orders.values())
    
    async def cancel_order(self, order_id: str) -> Dict[str, str]:
        """Cancel an order."""
        order = self._orders.get(order_id)
        if not order:
            return {"status": "error", "message": "Order not found"}
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return {"status": "error", "message": "Cannot cancel order in current status"}
        
        order.status = OrderStatus.CANCELLED
        await self.order_repository.update(order)
        
        return {"status": "success", "message": "Order cancelled successfully"}
