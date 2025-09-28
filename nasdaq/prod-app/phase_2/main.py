"""Main application entry point for Phase 2."""
import asyncio
from fastapi import FastAPI
from web.api.orders_api import router as orders_router
from gateways.fix_gateway.fix_server import FIXServer

app = FastAPI(
    title="NASDAQ Stock Market Simulator - Phase 2",
    description="Protocol Integration phase with FIX, FAST, and ITCH support",
    version="2.0.0"
)

# Include routers
app.include_router(orders_router)

# Protocol servers
fix_server = None


@app.on_event("startup")
async def startup_event():
    """Start protocol servers on application startup."""
    global fix_server
    
    # Start FIX server
    fix_server = FIXServer(port=9878)
    asyncio.create_task(fix_server.start())
    
    print("FIX protocol server started")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NASDAQ Stock Market Simulator - Phase 2",
        "phase": "Protocol Integration (Weeks 5-8)",
        "features": [
            "Enhanced FIX Gateway (Port 9878)",
            "Protocol message routing to OMS",
            "Session management",
            "Error handling and recovery"
        ],
        "endpoints": {
            "fix_gateway": "localhost:9878",
            "rest_api": "localhost:8000"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "phase": "2",
        "services": {
            "fix_server": fix_server.is_running if fix_server else False,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
