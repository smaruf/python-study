"""Market Data Engine for Phase 3."""
import asyncio
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketData:
    """Market data snapshot."""
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    last_price: float
    last_size: int
    timestamp: datetime


@dataclass
class Trade:
    """Trade data."""
    symbol: str
    price: float
    quantity: int
    timestamp: datetime
    trade_id: str


class MarketDataEngine:
    """Market Data Engine for Phase 3."""
    
    def __init__(self):
        self.subscriptions: Dict[str, List] = {}  # symbol -> list of subscribers
        self.market_data: Dict[str, MarketData] = {}
        self.trade_history: List[Trade] = []
        self.is_running = False
        
        # Initialize some symbols
        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"]
        self._initialize_market_data()
    
    def _initialize_market_data(self):
        """Initialize market data for symbols."""
        for symbol in self.symbols:
            base_price = random.uniform(100, 300)
            self.market_data[symbol] = MarketData(
                symbol=symbol,
                bid_price=base_price - 0.05,
                bid_size=random.randint(100, 1000),
                ask_price=base_price + 0.05,
                ask_size=random.randint(100, 1000),
                last_price=base_price,
                last_size=random.randint(10, 100),
                timestamp=datetime.utcnow()
            )
    
    async def start_market_simulation(self):
        """Start market data simulation."""
        self.is_running = True
        print("📈 Market Data Engine started")
        
        while self.is_running:
            # Update market data for all symbols
            for symbol in self.symbols:
                await self._update_symbol_data(symbol)
            
            await asyncio.sleep(1)  # Update every second
    
    async def _update_symbol_data(self, symbol: str):
        """Update market data for a symbol."""
        current = self.market_data[symbol]
        
        # Simulate price movement
        price_change = random.uniform(-0.5, 0.5)
        new_last_price = max(0.01, current.last_price + price_change)
        
        # Update market data
        current.last_price = new_last_price
        current.bid_price = new_last_price - random.uniform(0.01, 0.10)
        current.ask_price = new_last_price + random.uniform(0.01, 0.10)
        current.bid_size = random.randint(100, 1000)
        current.ask_size = random.randint(100, 1000)
        current.last_size = random.randint(10, 100)
        current.timestamp = datetime.utcnow()
        
        # Sometimes generate a trade
        if random.random() < 0.3:  # 30% chance
            trade = Trade(
                symbol=symbol,
                price=new_last_price,
                quantity=random.randint(10, 500),
                timestamp=datetime.utcnow(),
                trade_id=f"T_{datetime.utcnow().timestamp()}"
            )
            
            self.trade_history.append(trade)
            
            # Keep only recent trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-500:]
        
        # Notify subscribers
        await self._notify_subscribers(symbol, current)
    
    async def _notify_subscribers(self, symbol: str, data: MarketData):
        """Notify subscribers of market data updates."""
        if symbol in self.subscriptions:
            for subscriber in self.subscriptions[symbol]:
                try:
                    await subscriber(data)
                except Exception as e:
                    print(f"Error notifying subscriber: {e}")
    
    async def subscribe(self, symbol: str, callback):
        """Subscribe to market data for a symbol."""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        
        self.subscriptions[symbol].append(callback)
        print(f"📊 Subscribed to market data for {symbol}")
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol."""
        return self.market_data.get(symbol)
    
    async def get_all_market_data(self) -> Dict[str, MarketData]:
        """Get all market data."""
        return self.market_data.copy()
    
    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Trade]:
        """Get trade history."""
        trades = self.trade_history
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        return trades[-limit:]
    
    def stop(self):
        """Stop the market data engine."""
        self.is_running = False
        print("📈 Market Data Engine stopped")
