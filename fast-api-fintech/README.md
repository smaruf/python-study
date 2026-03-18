[← Back to Python Study Repository](../README.md)

# FastAPI Fintech Application

A comprehensive FastAPI-based fintech application supporting banking operations and stock exchange trading. This production-ready application provides REST APIs for managing bank accounts, transactions, stock trading, and market data.

## 🏦 Features

### Banking Module
- **Account Management**: Create, view, and manage bank accounts
- **Transactions**: Deposit, withdrawal, and transaction history
- **Transfers**: Money transfers between accounts
- **Balance Management**: Real-time balance tracking
- **Account Types**: Savings, Checking, Investment accounts

### Stock Exchange Module
- **Order Management**: Place, modify, and cancel stock orders
- **Market Data**: Real-time stock prices and market information
- **Trading**: Buy and sell stocks with different order types (Market, Limit, Stop)
- **Portfolio Management**: Track holdings and performance
- **Trade History**: Complete audit trail of all trades

### Security Features
- JWT-based authentication
- Role-based access control (RBAC)
- API rate limiting
- Request validation
- Secure password hashing

## 🏗️ Architecture

```
fast-api-fintech/
├── app/
│   ├── main.py                 # Application entry point
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/      # API route handlers
│   │       │   ├── banking.py  # Banking endpoints
│   │       │   ├── trading.py  # Stock exchange endpoints
│   │       │   ├── auth.py     # Authentication endpoints
│   │       │   └── users.py    # User management endpoints
│   │       └── api.py          # API router aggregation
│   ├── core/
│   │   ├── config.py           # Configuration management
│   │   ├── security.py         # Security utilities
│   │   └── dependencies.py     # Dependency injection
│   ├── models/                 # Database models (SQLAlchemy)
│   │   ├── user.py
│   │   ├── account.py
│   │   ├── transaction.py
│   │   ├── stock.py
│   │   └── order.py
│   ├── schemas/                # Pydantic schemas
│   │   ├── user.py
│   │   ├── account.py
│   │   ├── transaction.py
│   │   ├── stock.py
│   │   └── order.py
│   ├── services/               # Business logic
│   │   ├── banking_service.py
│   │   ├── trading_service.py
│   │   └── auth_service.py
│   ├── db/
│   │   ├── base.py            # Database base configuration
│   │   ├── session.py         # Database session management
│   │   └── init_db.py         # Database initialization
│   └── middleware/
│       ├── error_handler.py   # Global error handling
│       └── logging.py         # Request/response logging
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── scripts/
│   ├── init_db.sh            # Database initialization script
│   └── seed_data.py          # Seed sample data
├── docs/
│   ├── API.md                # API documentation
│   ├── DATABASE.md           # Database schema documentation
│   └── DEPLOYMENT.md         # Deployment guide
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ or SQLite (for development)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
cd fast-api-fintech
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize database**
```bash
python scripts/seed_data.py
```

6. **Run the application**
```bash
python -m app.main
# Or using uvicorn directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- API: http://localhost:8000
- Interactive API docs (Swagger): http://localhost:8000/docs
- Alternative API docs (ReDoc): http://localhost:8000/redoc

## 📚 API Documentation

### Authentication

#### Register a new user
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "John Doe"
  }'
```

#### Login
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'
```

Response includes JWT token:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Banking APIs

#### Create a bank account
```bash
curl -X POST "http://localhost:8000/api/v1/banking/accounts" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "account_type": "CHECKING",
    "currency": "USD",
    "initial_balance": 1000.00
  }'
```

#### Get account balance
```bash
curl -X GET "http://localhost:8000/api/v1/banking/accounts/{account_id}/balance" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Make a deposit
```bash
curl -X POST "http://localhost:8000/api/v1/banking/transactions/deposit" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "acc_123456",
    "amount": 500.00,
    "description": "Salary deposit"
  }'
```

#### Make a withdrawal
```bash
curl -X POST "http://localhost:8000/api/v1/banking/transactions/withdraw" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "acc_123456",
    "amount": 100.00,
    "description": "ATM withdrawal"
  }'
```

#### Transfer money between accounts
```bash
curl -X POST "http://localhost:8000/api/v1/banking/transactions/transfer" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "from_account_id": "acc_123456",
    "to_account_id": "acc_789012",
    "amount": 250.00,
    "description": "Payment to friend"
  }'
```

#### Get transaction history
```bash
curl -X GET "http://localhost:8000/api/v1/banking/transactions/history?account_id=acc_123456&limit=50" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Stock Exchange APIs

#### Get market data
```bash
curl -X GET "http://localhost:8000/api/v1/trading/market/stocks" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Get specific stock information
```bash
curl -X GET "http://localhost:8000/api/v1/trading/market/stocks/AAPL" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Place a buy order
```bash
curl -X POST "http://localhost:8000/api/v1/trading/orders" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "order_type": "LIMIT",
    "side": "BUY",
    "quantity": 10,
    "price": 150.00
  }'
```

#### Place a sell order
```bash
curl -X POST "http://localhost:8000/api/v1/trading/orders" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "order_type": "MARKET",
    "side": "SELL",
    "quantity": 5
  }'
```

#### Get order status
```bash
curl -X GET "http://localhost:8000/api/v1/trading/orders/{order_id}" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Cancel an order
```bash
curl -X DELETE "http://localhost:8000/api/v1/trading/orders/{order_id}" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Get portfolio
```bash
curl -X GET "http://localhost:8000/api/v1/trading/portfolio" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Get trade history
```bash
curl -X GET "http://localhost:8000/api/v1/trading/trades/history?limit=50" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 🗄️ Database Schema

### Core Tables

#### Users
- `id`: UUID (Primary Key)
- `email`: String (Unique)
- `password_hash`: String
- `full_name`: String
- `is_active`: Boolean
- `role`: Enum (USER, ADMIN)
- `created_at`: Timestamp
- `updated_at`: Timestamp

#### Accounts
- `id`: UUID (Primary Key)
- `user_id`: UUID (Foreign Key → Users)
- `account_number`: String (Unique)
- `account_type`: Enum (CHECKING, SAVINGS, INVESTMENT)
- `balance`: Decimal
- `currency`: String (Default: USD)
- `status`: Enum (ACTIVE, FROZEN, CLOSED)
- `created_at`: Timestamp
- `updated_at`: Timestamp

#### Transactions
- `id`: UUID (Primary Key)
- `account_id`: UUID (Foreign Key → Accounts)
- `transaction_type`: Enum (DEPOSIT, WITHDRAWAL, TRANSFER)
- `amount`: Decimal
- `balance_after`: Decimal
- `description`: String
- `status`: Enum (PENDING, COMPLETED, FAILED)
- `created_at`: Timestamp

#### Stocks
- `id`: UUID (Primary Key)
- `symbol`: String (Unique)
- `name`: String
- `current_price`: Decimal
- `previous_close`: Decimal
- `market_cap`: Decimal
- `volume`: Integer
- `updated_at`: Timestamp

#### Orders
- `id`: UUID (Primary Key)
- `user_id`: UUID (Foreign Key → Users)
- `symbol`: String
- `order_type`: Enum (MARKET, LIMIT, STOP)
- `side`: Enum (BUY, SELL)
- `quantity`: Integer
- `price`: Decimal (Optional for MARKET orders)
- `status`: Enum (PENDING, FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED)
- `filled_quantity`: Integer
- `average_price`: Decimal
- `created_at`: Timestamp
- `updated_at`: Timestamp

## 🔒 Security

### Authentication
- JWT (JSON Web Tokens) for stateless authentication
- Access tokens with configurable expiration
- Refresh token support
- Password hashing using bcrypt

### Authorization
- Role-based access control (RBAC)
- User and Admin roles
- Protected endpoints with permission checks

### Best Practices
- Input validation using Pydantic
- SQL injection prevention via SQLAlchemy ORM
- CORS configuration
- Rate limiting on sensitive endpoints
- Secure password requirements

## 🧪 Testing

### Run unit tests
```bash
pytest tests/unit -v
```

### Run integration tests
```bash
pytest tests/integration -v
```

### Run all tests with coverage
```bash
pytest --cov=app --cov-report=html
```

## 📊 Monitoring and Logging

- Structured logging with correlation IDs
- Request/response logging
- Error tracking
- Performance metrics
- Health check endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

## 🚀 Deployment

### Docker Deployment
```bash
docker build -t fastapi-fintech .
docker run -p 8000:8000 --env-file .env fastapi-fintech
```

### Production Considerations
- Use PostgreSQL for production database
- Configure environment variables properly
- Enable HTTPS/TLS
- Set up load balancing
- Configure monitoring and alerting
- Regular database backups
- Use gunicorn or uvicorn with multiple workers

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment instructions.

## 🛠️ Development

### Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Use isort for import sorting
- Type hints for all functions

### Format code
```bash
black app/ tests/
isort app/ tests/
```

### Lint code
```bash
flake8 app/ tests/
mypy app/
```

## 📝 Environment Variables

Key environment variables (see `.env.example`):

```bash
# Application
APP_NAME=FastAPI Fintech
DEBUG=False
API_V1_PREFIX=/api/v1

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/fintech_db

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256

# CORS
BACKEND_CORS_ORIGINS=["http://localhost:3000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

## 🔗 Related Documentation

- [API Documentation](docs/API.md)
- [Database Schema](docs/DATABASE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://www.sqlalchemy.org/)

## 🎯 Roadmap

### Current Features (v1.0)
- ✅ Basic banking operations
- ✅ Stock trading functionality
- ✅ JWT authentication
- ✅ REST API

### Planned Features (v2.0)
- [ ] WebSocket support for real-time market data
- [ ] Advanced order types (Stop-Limit, Trailing Stop)
- [ ] Multi-currency support
- [ ] Transaction notifications
- [ ] Scheduled payments
- [ ] Credit/Debit card integration
- [ ] Mobile app support
- [ ] Analytics dashboard
- [ ] Compliance and reporting tools
- [ ] Loan management

## 📚 Additional Resources

- **Fintech Regulations**: Ensure compliance with local financial regulations
- **Security Best Practices**: Follow OWASP guidelines
- **Performance Optimization**: Use caching, connection pooling, and async operations
- **Testing**: Maintain high test coverage (>80%)
