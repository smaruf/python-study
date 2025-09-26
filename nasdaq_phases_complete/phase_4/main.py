"""Main application entry point for Phase 4."""
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from web.api.orders_api import router as orders_router
from engines.risk_engine.risk_service import RiskEngine
from engines.market_data_engine.market_data_service import MarketDataEngine
from engines.settlement_engine.settlement_service import SettlementEngine
from engines.analytics_engine.analytics_service import AnalyticsEngine

app = FastAPI(
    title="NASDAQ Stock Market Simulator - Phase 4",
    description="Complete trading lifecycle with analytics and dashboard",
    version="4.0.0"
)

# Include routers
app.include_router(orders_router)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="web/ui/static"), name="static")
templates = Jinja2Templates(directory="web/ui/templates")

# Engines
risk_engine = None
market_data_engine = None
settlement_engine = None
analytics_engine = None


@app.on_event("startup")
async def startup_event():
    """Start all engines on application startup."""
    global risk_engine, market_data_engine, settlement_engine, analytics_engine
    
    # Initialize engines
    risk_engine = RiskEngine()
    market_data_engine = MarketDataEngine()
    settlement_engine = SettlementEngine()
    analytics_engine = AnalyticsEngine()
    
    # Start background services
    asyncio.create_task(market_data_engine.start_market_simulation())
    asyncio.create_task(settlement_processing_loop())
    
    print("All engines started - Production ready!")


async def settlement_processing_loop():
    """Background settlement processing."""
    while True:
        if settlement_engine:
            await settlement_engine.process_settlements()
        await asyncio.sleep(60)  # Process every minute


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary for dashboard."""
    if analytics_engine:
        metrics = await analytics_engine.generate_trading_metrics()
        market_summary = await analytics_engine.get_market_summary()
        
        return {
            "total_orders": metrics.total_orders,
            "fill_rate": metrics.fill_rate * 100,
            "total_volume": metrics.total_volume,
            "active_symbols": len(market_summary),
            "market_summary": market_summary
        }
    return {}


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


@app.get("/settlement/summary")
async def get_settlement_summary():
    """Get settlement summary."""
    if settlement_engine:
        return await settlement_engine.get_settlement_summary()
    return {}


@app.get("/reports/{symbol}")
async def generate_report(symbol: str, period_hours: int = 24):
    """Generate performance report."""
    if analytics_engine:
        report = await analytics_engine.generate_performance_report(symbol, period_hours)
        return {
            "report_id": report.report_id,
            "symbol": report.symbol,
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "metrics": {
                "total_orders": report.metrics.total_orders,
                "filled_orders": report.metrics.filled_orders,
                "fill_rate": round(report.metrics.fill_rate * 100, 2),
                "total_volume": report.metrics.total_volume,
                "total_value": report.metrics.total_value,
                "average_fill_price": round(report.metrics.average_fill_price, 2)
            }
        }
    return {}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "phase": "4",
        "engines": {
            "risk_engine": risk_engine is not None,
            "market_data_engine": market_data_engine is not None and market_data_engine.is_running,
            "settlement_engine": settlement_engine is not None,
            "analytics_engine": analytics_engine is not None
        },
        "production_ready": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
