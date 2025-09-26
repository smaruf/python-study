"""Settlement Engine for Phase 4."""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class SettlementStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SETTLED = "SETTLED"
    FAILED = "FAILED"


@dataclass
class Settlement:
    """Settlement record."""
    settlement_id: str
    trade_id: str
    symbol: str
    quantity: int
    price: float
    settlement_date: datetime
    status: SettlementStatus
    counterparty: str
    created_at: datetime


class SettlementEngine:
    """Settlement Engine for Phase 4."""
    
    def __init__(self):
        self.settlements: Dict[str, Settlement] = {}
        self.settlement_rules = {
            "T+2": 2,  # Standard settlement is T+2
            "T+1": 1,  # Some instruments settle T+1
            "T+0": 0   # Same day settlement
        }
    
    async def create_settlement(self, trade_id: str, symbol: str, quantity: int, 
                              price: float, counterparty: str) -> Settlement:
        """Create a new settlement."""
        settlement_date = datetime.utcnow() + timedelta(days=self.settlement_rules["T+2"])
        
        settlement = Settlement(
            settlement_id=f"SETTLE_{datetime.utcnow().timestamp()}",
            trade_id=trade_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            settlement_date=settlement_date,
            status=SettlementStatus.PENDING,
            counterparty=counterparty,
            created_at=datetime.utcnow()
        )
        
        self.settlements[settlement.settlement_id] = settlement
        return settlement
    
    async def process_settlements(self):
        """Process pending settlements."""
        current_time = datetime.utcnow()
        
        for settlement in self.settlements.values():
            if (settlement.status == SettlementStatus.PENDING and 
                settlement.settlement_date <= current_time):
                
                # Simulate settlement processing
                settlement.status = SettlementStatus.PROCESSING
                
                # In a real system, this would involve:
                # - DVP (Delivery vs Payment) processing
                # - Clearing house integration
                # - Cash and security transfers
                
                # For simulation, randomly succeed or fail
                import random
                if random.random() > 0.05:  # 95% success rate
                    settlement.status = SettlementStatus.SETTLED
                else:
                    settlement.status = SettlementStatus.FAILED
                
                print(f"Settlement {settlement.settlement_id}: {settlement.status.value}")
    
    async def get_settlement_summary(self) -> Dict:
        """Get settlement summary."""
        total = len(self.settlements)
        by_status = {}
        
        for status in SettlementStatus:
            by_status[status.value] = len([s for s in self.settlements.values() if s.status == status])
        
        return {
            "total_settlements": total,
            "by_status": by_status,
            "settlement_rules": self.settlement_rules
        }
