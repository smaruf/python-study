"""Main application entry point for Phase 1."""
from fastapi import FastAPI
from web.api.orders_api import router as orders_router

app = FastAPI(
    title="NASDAQ Stock Market Simulator - Phase 1",
    description="Foundation phase with basic OMS functionality",
    version="1.0.0"
)

# Include routers
app.include_router(orders_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NASDAQ Stock Market Simulator - Phase 1",
        "phase": "Foundation (Weeks 1-4)",
        "features": [
            "Basic Order Management System",
            "Order submission and tracking",
            "REST API endpoints",
            "In-memory data storage"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "phase": "1"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
