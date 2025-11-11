"""API v1 router aggregation."""
from fastapi import APIRouter
from app.api.v1.endpoints import auth, users, banking, trading

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(banking.router)
api_router.include_router(trading.router)
