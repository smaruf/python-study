"""User management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.user import User, UserUpdate
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    from datetime import datetime
    
    return {
        "id": current_user["user_id"],
        "email": current_user["email"],
        "full_name": "John Doe",  # Mock data
        "is_active": True,
        "role": current_user["role"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@router.put("/me", response_model=User)
async def update_current_user(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update current user information.
    """
    from datetime import datetime
    
    # In production, update database
    return {
        "id": current_user["user_id"],
        "email": user_update.email or current_user["email"],
        "full_name": user_update.full_name or "John Doe",
        "is_active": True,
        "role": current_user["role"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@router.get("/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user by ID.
    
    Requires authentication.
    """
    from datetime import datetime
    
    # In production, fetch from database
    return {
        "id": user_id,
        "email": "user@example.com",
        "full_name": "John Doe",
        "is_active": True,
        "role": "user",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
