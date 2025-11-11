"""Authentication endpoints."""
from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.user import UserCreate, User, Token
from app.core.security import verify_password, get_password_hash, create_access_token, create_refresh_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user.
    
    - **email**: Valid email address
    - **password**: Minimum 8 characters
    - **full_name**: User's full name
    """
    # In production, this would interact with a database
    # For now, returning mock response
    import uuid
    from datetime import datetime
    
    return {
        "id": str(uuid.uuid4()),
        "email": user_data.email,
        "full_name": user_data.full_name,
        "is_active": True,
        "role": "user",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@router.post("/login", response_model=Token)
async def login(email: str, password: str):
    """
    Login with email and password.
    
    Returns JWT access token and refresh token.
    """
    # In production, verify against database
    # For now, creating mock token
    token_data = {
        "sub": "user_123",
        "email": email,
        "role": "user"
    }
    
    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token.
    """
    from app.core.security import decode_token
    
    payload = decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    token_data = {
        "sub": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role")
    }
    
    new_access_token = create_access_token(data=token_data)
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }


@router.post("/logout")
async def logout():
    """
    Logout current user.
    
    In production, this would invalidate the token.
    """
    return {"message": "Successfully logged out"}
