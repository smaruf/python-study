"""Seed initial data for the application."""
import asyncio
from decimal import Decimal
from datetime import datetime

# Mock data seeding script
# In production, this would interact with actual database

async def seed_stocks():
    """Seed stock market data."""
    stocks = [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "current_price": Decimal("175.50"),
            "previous_close": Decimal("173.25"),
            "market_cap": Decimal("2750000000000"),
            "volume": 45000000
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "current_price": Decimal("140.25"),
            "previous_close": Decimal("139.50"),
            "market_cap": Decimal("1750000000000"),
            "volume": 28000000
        },
        {
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "current_price": Decimal("385.75"),
            "previous_close": Decimal("383.00"),
            "market_cap": Decimal("2850000000000"),
            "volume": 32000000
        },
        {
            "symbol": "AMZN",
            "name": "Amazon.com Inc.",
            "current_price": Decimal("145.80"),
            "previous_close": Decimal("144.20"),
            "market_cap": Decimal("1500000000000"),
            "volume": 35000000
        },
        {
            "symbol": "TSLA",
            "name": "Tesla Inc.",
            "current_price": Decimal("248.50"),
            "previous_close": Decimal("245.00"),
            "market_cap": Decimal("780000000000"),
            "volume": 52000000
        }
    ]
    
    print("📊 Seeding stock market data...")
    for stock in stocks:
        print(f"  ✓ {stock['symbol']}: {stock['name']}")
    
    return stocks


async def seed_users():
    """Seed demo users."""
    users = [
        {
            "email": "demo@fintech.com",
            "password": "Demo123456!",
            "full_name": "Demo User",
            "role": "user"
        },
        {
            "email": "admin@fintech.com",
            "password": "Admin123456!",
            "full_name": "Admin User",
            "role": "admin"
        }
    ]
    
    print("👥 Seeding demo users...")
    for user in users:
        print(f"  ✓ {user['email']} ({user['role']})")
    
    return users


async def main():
    """Main seeding function."""
    print("🌱 Starting database seeding...")
    print()
    
    try:
        # Seed data
        await seed_stocks()
        print()
        await seed_users()
        print()
        
        print("✅ Database seeding completed successfully!")
        print()
        print("📝 Demo Credentials:")
        print("   User:  demo@fintech.com / Demo123456!")
        print("   Admin: admin@fintech.com / Admin123456!")
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
