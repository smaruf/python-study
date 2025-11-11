"""Application configuration."""
from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "FastAPI Fintech Application"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./fintech.db"
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Trading Configuration
    MARKET_OPEN_HOUR: int = 9
    MARKET_CLOSE_HOUR: int = 16
    ENABLE_AFTER_HOURS_TRADING: bool = False
    
    # Banking Configuration
    MIN_ACCOUNT_BALANCE: float = 0.00
    MAX_DAILY_TRANSFER_LIMIT: float = 10000.00
    CURRENCY_DEFAULT: str = "USD"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
