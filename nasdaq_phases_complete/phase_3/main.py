"""Main application entry point for Phase 3."""
import asyncio
from fastapi import FastAPI
from web.api.orders_api import router as orders_router
from engines.risk_engine.risk_service import RiskEngine
from engines.market_data_engine.market_data_service import MarketDataEngine

app = FastAPI(
    title="NASDAQ Stock Market Simulator - Phase 3",
    description="Market Data and Risk Management phase",
    version="3.0.0"
)

# Include routers
app.include_router(orders_router)

# Engines
risk_engine = None
market_data_engine = None


@app.on_event("startup")
async def startup_event():
    """Start engines on application startup."""
    global risk_engine, market_data_engine
    
    # Initialize engines
    risk_engine = RiskEngine()
    market_data_engine = MarketDataEngine()
    
    # Start market data simulation
    asyncio.create_task(market_data_engine.start_market_simulation())
    
    print("Risk Engine and Market Data Engine started")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NASDAQ Stock Market Simulator - Phase 3",
        "phase": "Market Data and Risk (Weeks 9-12)",
        "features": [
            "Real-time Market Data Engine",
            "Risk Management Engine",
            "Pre-trade risk checks",
            "Position tracking",
            "Market data simulation",
            "Risk alerts and monitoring"
        ]
    }


@app.get("/market-data")
async def get_market_data():
    """Get all current market data."""
    if market_data_engine:
        data = await market_data_engine.get_all_market_data()
        return {symbol: {
            "symbol": md.symbol,
            "bid_price": md.bid_price,
            "bid_size": md.bid_size,
            "ask_price": md.ask_price,
            "ask_size": md.ask_size,
            "last_price": md.last_price,
            "last_size": md.last_size,
            "timestamp": md.timestamp.isoformat()
        } for symbol, md in data.items()}
    return {}


@app.get("/market-data/{symbol}")
async def get_symbol_market_data(symbol: str):
    """Get market data for specific symbol."""
    if market_data_engine:
        data = await market_data_engine.get_market_data(symbol)
        if data:
            return {
                "symbol": data.symbol,
                "bid_price": data.bid_price,
                "bid_size": data.bid_size,
                "ask_price": data.ask_price,
                "ask_size": data.ask_size,
                "last_price": data.last_price,
                "last_size": data.last_size,
                "timestamp": data.timestamp.isoformat()
            }
    return {"error": "Symbol not found"}


@app.get("/risk/summary")
async def get_risk_summary():
    """Get risk management summary."""
    if risk_engine:
        return await risk_engine.get_risk_summary()
    return {}


@app.get("/trades")
async def get_trades(symbol: str = None, limit: int = 100):
    """Get trade history."""
    if market_data_engine:
        trades = await market_data_engine.get_trade_history(symbol, limit)
        return [{
            "symbol": t.symbol,
            "price": t.price,
            "quantity": t.quantity,
            "timestamp": t.timestamp.isoformat(),
            "trade_id": t.trade_id
        } for t in trades]
    return []


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "phase": "3",
        "engines": {
            "risk_engine": risk_engine is not None,
            "market_data_engine": market_data_engine is not None and market_data_engine.is_running
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
