"""Risk Management Engine for Phase 3."""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    limit_type: str
    symbol: Optional[str]
    max_quantity: int
    max_value: float
    max_orders: int


@dataclass
class RiskAlert:
    """Risk alert information."""
    alert_id: str
    risk_level: RiskLevel
    message: str
    symbol: Optional[str]
    created_at: datetime


class RiskEngine:
    """Risk Management Engine for Phase 3."""
    
    def __init__(self):
        self.limits: Dict[str, RiskLimit] = {}
        self.alerts: List[RiskAlert] = []
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        
        # Initialize default limits
        self._setup_default_limits()
    
    def _setup_default_limits(self):
        """Setup default risk limits."""
        self.limits["global_max_orders"] = RiskLimit(
            limit_type="MAX_ORDERS",
            symbol=None,
            max_quantity=0,
            max_value=0,
            max_orders=1000
        )
        
        self.limits["single_order_max"] = RiskLimit(
            limit_type="ORDER_SIZE",
            symbol=None,
            max_quantity=10000,
            max_value=1000000,
            max_orders=0
        )
    
    async def check_pre_trade_risk(self, order) -> Dict[str, any]:
        """Perform pre-trade risk checks."""
        try:
            # Check order size limits
            if order.quantity > self.limits["single_order_max"].max_quantity:
                return {
                    "status": "rejected",
                    "reason": "Order quantity exceeds maximum limit",
                    "risk_level": RiskLevel.HIGH.value
                }
            
            # Check order value limits
            if order.price and (order.quantity * order.price) > self.limits["single_order_max"].max_value:
                return {
                    "status": "rejected", 
                    "reason": "Order value exceeds maximum limit",
                    "risk_level": RiskLevel.HIGH.value
                }
            
            # Check position limits (simplified)
            current_position = self.positions.get(order.symbol, {"quantity": 0})
            new_position = current_position["quantity"]
            
            if order.side.value == "BUY":
                new_position += order.quantity
            else:
                new_position -= order.quantity
            
            if abs(new_position) > 50000:  # Max position limit
                return {
                    "status": "rejected",
                    "reason": "Position limit would be exceeded",
                    "risk_level": RiskLevel.MEDIUM.value
                }
            
            return {
                "status": "approved",
                "risk_level": RiskLevel.LOW.value,
                "message": "Order passed all risk checks"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "reason": f"Risk check failed: {str(e)}",
                "risk_level": RiskLevel.CRITICAL.value
            }
    
    async def update_position(self, symbol: str, quantity: int, price: float):
        """Update position after trade execution."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0,
                "average_price": 0,
                "total_value": 0
            }
        
        current = self.positions[symbol]
        current["quantity"] += quantity
        current["total_value"] += quantity * price
        
        if current["quantity"] != 0:
            current["average_price"] = current["total_value"] / current["quantity"]
    
    async def generate_risk_alert(self, risk_level: RiskLevel, message: str, symbol: str = None):
        """Generate a risk alert."""
        alert = RiskAlert(
            alert_id=f"RISK_{datetime.utcnow().timestamp()}",
            risk_level=risk_level,
            message=message,
            symbol=symbol,
            created_at=datetime.utcnow()
        )
        
        self.alerts.append(alert)
        print(f"🚨 Risk Alert: {alert.risk_level.value} - {alert.message}")
    
    async def get_risk_summary(self) -> Dict:
        """Get risk summary."""
        return {
            "total_positions": len(self.positions),
            "active_alerts": len([a for a in self.alerts if a.created_at > datetime.utcnow().replace(hour=0, minute=0, second=0)]),
            "risk_limits": len(self.limits),
            "positions": self.positions
        }
