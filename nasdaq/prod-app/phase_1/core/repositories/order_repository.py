"""Order repository for Phase 1."""
from typing import List, Optional
from core.models.order import Order


class OrderRepository:
    """Basic in-memory order repository for Phase 1."""
    
    def __init__(self):
        self._orders = {}
    
    async def save(self, order: Order) -> None:
        """Save an order."""
        self._orders[order.order_id] = order
    
    async def get_by_id(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    async def get_all(self) -> List[Order]:
        """Get all orders."""
        return list(self._orders.values())
    
    async def update(self, order: Order) -> None:
        """Update an order."""
        self._orders[order.order_id] = order
    
    async def delete(self, order_id: str) -> None:
        """Delete an order."""
        if order_id in self._orders:
            del self._orders[order_id]
