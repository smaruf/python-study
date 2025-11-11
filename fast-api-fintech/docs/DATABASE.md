# Database Schema Documentation

This document describes the database schema for the FastAPI Fintech Application.

## Overview

The application uses a relational database structure with the following main entities:

- **Users**: System users and authentication
- **Accounts**: Bank accounts owned by users
- **Transactions**: Banking transactions (deposits, withdrawals, transfers)
- **Stocks**: Available stocks for trading
- **Orders**: Stock trading orders
- **Trades**: Executed trades

## Entity Relationship Diagram

```
┌─────────────┐
│    Users    │
└──────┬──────┘
       │ 1
       │
       │ *
┌──────┴──────┐          ┌──────────────┐
│  Accounts   │ 1     *  │ Transactions │
└──────┬──────┘──────────└──────────────┘
       │ 1
       │
       │ *
┌──────┴──────┐          ┌──────────┐
│   Orders    │──────────│  Stocks  │
└──────┬──────┘ *     1  └──────────┘
       │ 1
       │
       │ *
┌──────┴──────┐
│   Trades    │
└─────────────┘
```

## Tables

### Users Table

Stores user account information and credentials.

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_active ON users(is_active);
```

**Fields:**
- `id`: Unique identifier (UUID)
- `email`: User's email address (unique)
- `password_hash`: Hashed password using bcrypt
- `full_name`: User's full name
- `is_active`: Account status flag
- `role`: User role (user, admin)
- `created_at`: Account creation timestamp
- `updated_at`: Last update timestamp

### Accounts Table

Stores bank account information.

```sql
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_number VARCHAR(50) UNIQUE NOT NULL,
    account_type VARCHAR(50) NOT NULL,
    balance DECIMAL(20, 2) DEFAULT 0.00,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(50) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT account_type_check CHECK (account_type IN ('CHECKING', 'SAVINGS', 'INVESTMENT')),
    CONSTRAINT account_status_check CHECK (status IN ('ACTIVE', 'FROZEN', 'CLOSED')),
    CONSTRAINT balance_non_negative CHECK (balance >= 0)
);

CREATE INDEX idx_accounts_user_id ON accounts(user_id);
CREATE INDEX idx_accounts_account_number ON accounts(account_number);
CREATE INDEX idx_accounts_status ON accounts(status);
```

**Fields:**
- `id`: Unique identifier (UUID)
- `user_id`: Foreign key to users table
- `account_number`: Unique account number
- `account_type`: Type of account (CHECKING, SAVINGS, INVESTMENT)
- `balance`: Current account balance
- `currency`: Currency code (e.g., USD, EUR)
- `status`: Account status (ACTIVE, FROZEN, CLOSED)
- `created_at`: Account creation timestamp
- `updated_at`: Last update timestamp

### Transactions Table

Stores all banking transactions.

```sql
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    transaction_type VARCHAR(50) NOT NULL,
    amount DECIMAL(20, 2) NOT NULL,
    balance_after DECIMAL(20, 2) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'COMPLETED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT transaction_type_check CHECK (transaction_type IN ('DEPOSIT', 'WITHDRAWAL', 'TRANSFER')),
    CONSTRAINT transaction_status_check CHECK (status IN ('PENDING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    CONSTRAINT amount_positive CHECK (amount > 0)
);

CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
CREATE INDEX idx_transactions_status ON transactions(status);
```

**Fields:**
- `id`: Unique identifier (UUID)
- `account_id`: Foreign key to accounts table
- `transaction_type`: Type of transaction (DEPOSIT, WITHDRAWAL, TRANSFER)
- `amount`: Transaction amount
- `balance_after`: Account balance after transaction
- `description`: Optional description
- `status`: Transaction status (PENDING, COMPLETED, FAILED, CANCELLED)
- `created_at`: Transaction timestamp

### Stocks Table

Stores information about available stocks.

```sql
CREATE TABLE stocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    current_price DECIMAL(20, 2) NOT NULL,
    previous_close DECIMAL(20, 2) NOT NULL,
    market_cap DECIMAL(30, 2),
    volume BIGINT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT price_positive CHECK (current_price > 0 AND previous_close > 0),
    CONSTRAINT volume_non_negative CHECK (volume >= 0)
);

CREATE UNIQUE INDEX idx_stocks_symbol ON stocks(symbol);
CREATE INDEX idx_stocks_updated_at ON stocks(updated_at DESC);
```

**Fields:**
- `id`: Unique identifier (UUID)
- `symbol`: Stock ticker symbol (unique)
- `name`: Company name
- `current_price`: Current stock price
- `previous_close`: Previous closing price
- `market_cap`: Market capitalization
- `volume`: Trading volume
- `updated_at`: Last price update timestamp

### Orders Table

Stores stock trading orders.

```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    order_type VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(20, 2),
    status VARCHAR(50) DEFAULT 'PENDING',
    filled_quantity INTEGER DEFAULT 0,
    average_price DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT order_type_check CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')),
    CONSTRAINT order_side_check CHECK (side IN ('BUY', 'SELL')),
    CONSTRAINT order_status_check CHECK (status IN ('PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED')),
    CONSTRAINT quantity_positive CHECK (quantity > 0),
    CONSTRAINT filled_quantity_valid CHECK (filled_quantity >= 0 AND filled_quantity <= quantity),
    CONSTRAINT price_positive CHECK (price IS NULL OR price > 0)
);

CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
```

**Fields:**
- `id`: Unique identifier (UUID)
- `user_id`: Foreign key to users table
- `symbol`: Stock ticker symbol
- `order_type`: Type of order (MARKET, LIMIT, STOP, STOP_LIMIT)
- `side`: Buy or sell (BUY, SELL)
- `quantity`: Number of shares
- `price`: Price per share (optional for MARKET orders)
- `status`: Order status (PENDING, FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED)
- `filled_quantity`: Number of shares filled
- `average_price`: Average execution price
- `created_at`: Order creation timestamp
- `updated_at`: Last update timestamp

### Trades Table

Stores executed trades.

```sql
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(20, 2) NOT NULL,
    total_value DECIMAL(20, 2) NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT trade_side_check CHECK (side IN ('BUY', 'SELL')),
    CONSTRAINT trade_quantity_positive CHECK (quantity > 0),
    CONSTRAINT trade_price_positive CHECK (price > 0)
);

CREATE INDEX idx_trades_order_id ON trades(order_id);
CREATE INDEX idx_trades_user_id ON trades(user_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_executed_at ON trades(executed_at DESC);
```

**Fields:**
- `id`: Unique identifier (UUID)
- `order_id`: Foreign key to orders table
- `user_id`: Foreign key to users table
- `symbol`: Stock ticker symbol
- `side`: Buy or sell (BUY, SELL)
- `quantity`: Number of shares traded
- `price`: Execution price per share
- `total_value`: Total trade value
- `executed_at`: Execution timestamp

## Indexes

Indexes are created on frequently queried columns to improve performance:

- User email and active status
- Account numbers and user relationships
- Transaction account relationships and timestamps
- Stock symbols
- Order user relationships, symbols, and status
- Trade relationships and timestamps

## Constraints

### Foreign Keys
- Accounts reference Users
- Transactions reference Accounts
- Orders reference Users
- Trades reference Orders and Users

### Check Constraints
- Email format validation
- Account types and statuses
- Non-negative balances
- Positive transaction amounts
- Valid order types and statuses
- Positive prices and quantities

## Database Migrations

Use Alembic for database migrations:

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

## Performance Considerations

1. **Indexes**: Created on frequently queried columns
2. **Partitioning**: Consider partitioning transactions and trades tables by date for large datasets
3. **Archiving**: Implement data archiving for old transactions
4. **Connection Pooling**: Use SQLAlchemy connection pooling
5. **Query Optimization**: Use appropriate joins and avoid N+1 queries

## Backup and Recovery

1. **Regular Backups**: Scheduled daily backups
2. **Point-in-Time Recovery**: Enable WAL archiving for PostgreSQL
3. **Replication**: Consider primary-replica setup for high availability
4. **Disaster Recovery**: Maintain offsite backups

## Security

1. **Password Hashing**: Using bcrypt with appropriate cost factor
2. **SQL Injection**: Protected via SQLAlchemy ORM
3. **Data Encryption**: Consider encrypting sensitive fields at rest
4. **Access Control**: Row-level security for multi-tenant data
5. **Audit Logging**: Track all data modifications

## Data Retention

1. **Transactions**: Keep all transaction history indefinitely for audit purposes
2. **Trades**: Archive trades older than 7 years
3. **Orders**: Archive filled/cancelled orders older than 1 year
4. **Logs**: Retain application logs for 90 days
