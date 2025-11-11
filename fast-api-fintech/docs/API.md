# API Documentation

Complete API reference for the FastAPI Fintech Application.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All protected endpoints require a Bearer token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## Rate Limits

- 60 requests per minute per user
- Rate limit headers are included in responses

## Response Format

All API responses follow this standard format:

### Success Response
```json
{
  "data": { ... },
  "message": "Success"
}
```

### Error Response
```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## HTTP Status Codes

- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `204 No Content` - Request succeeded with no response body
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required or failed
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Authentication Endpoints

### Register

Create a new user account.

**Endpoint:** `POST /api/v1/auth/register`

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "full_name": "John Doe"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "role": "user",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

### Login

Authenticate and receive JWT tokens.

**Endpoint:** `POST /api/v1/auth/login`

**Query Parameters:**
- `email` (string, required)
- `password` (string, required)

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

### Refresh Token

Get a new access token using refresh token.

**Endpoint:** `POST /api/v1/auth/refresh`

**Query Parameters:**
- `refresh_token` (string, required)

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

## Banking Endpoints

### Create Account

Create a new bank account.

**Endpoint:** `POST /api/v1/banking/accounts`

**Authentication:** Required

**Request Body:**
```json
{
  "account_type": "CHECKING",
  "currency": "USD",
  "initial_balance": 1000.00
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "account_number": "ACC1234567890",
  "account_type": "CHECKING",
  "balance": 1000.00,
  "currency": "USD",
  "status": "ACTIVE",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

### Get Accounts

List all accounts for the authenticated user.

**Endpoint:** `GET /api/v1/banking/accounts`

**Authentication:** Required

**Query Parameters:**
- `status_filter` (optional): Filter by status (ACTIVE, FROZEN, CLOSED)

**Response:** `200 OK`
```json
[
  {
    "id": "uuid",
    "user_id": "uuid",
    "account_number": "ACC1234567890",
    "account_type": "CHECKING",
    "balance": 5000.00,
    "currency": "USD",
    "status": "ACTIVE",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00"
  }
]
```

### Deposit

Deposit money into an account.

**Endpoint:** `POST /api/v1/banking/transactions/deposit`

**Authentication:** Required

**Request Body:**
```json
{
  "account_id": "uuid",
  "amount": 500.00,
  "description": "Salary deposit"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "account_id": "uuid",
  "transaction_type": "DEPOSIT",
  "amount": 500.00,
  "balance_after": 5500.00,
  "description": "Salary deposit",
  "status": "COMPLETED",
  "created_at": "2024-01-01T00:00:00"
}
```

### Withdrawal

Withdraw money from an account.

**Endpoint:** `POST /api/v1/banking/transactions/withdraw`

**Authentication:** Required

**Request Body:**
```json
{
  "account_id": "uuid",
  "amount": 100.00,
  "description": "ATM withdrawal"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "account_id": "uuid",
  "transaction_type": "WITHDRAWAL",
  "amount": 100.00,
  "balance_after": 4900.00,
  "description": "ATM withdrawal",
  "status": "COMPLETED",
  "created_at": "2024-01-01T00:00:00"
}
```

### Transfer

Transfer money between accounts.

**Endpoint:** `POST /api/v1/banking/transactions/transfer`

**Authentication:** Required

**Request Body:**
```json
{
  "from_account_id": "uuid",
  "to_account_id": "uuid",
  "amount": 250.00,
  "description": "Payment to friend"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "account_id": "uuid",
  "transaction_type": "TRANSFER",
  "amount": 250.00,
  "balance_after": 4750.00,
  "description": "Payment to friend",
  "status": "COMPLETED",
  "created_at": "2024-01-01T00:00:00"
}
```

## Trading Endpoints

### Get Market Stocks

List available stocks with current prices.

**Endpoint:** `GET /api/v1/trading/market/stocks`

**Authentication:** Required

**Query Parameters:**
- `limit` (optional, default: 50, max: 100)
- `offset` (optional, default: 0)

**Response:** `200 OK`
```json
{
  "stocks": [
    {
      "id": "uuid",
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "current_price": 175.50,
      "previous_close": 173.25,
      "market_cap": 2750000000000,
      "volume": 45000000,
      "updated_at": "2024-01-01T00:00:00",
      "change": 2.25,
      "change_percent": 1.30
    }
  ],
  "total": 50
}
```

### Place Order

Place a new stock order.

**Endpoint:** `POST /api/v1/trading/orders`

**Authentication:** Required

**Request Body:**
```json
{
  "symbol": "AAPL",
  "order_type": "LIMIT",
  "side": "BUY",
  "quantity": 10,
  "price": 175.00
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "symbol": "AAPL",
  "order_type": "LIMIT",
  "side": "BUY",
  "quantity": 10,
  "price": 175.00,
  "status": "PENDING",
  "filled_quantity": 0,
  "average_price": null,
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

### Get Orders

List all orders for the authenticated user.

**Endpoint:** `GET /api/v1/trading/orders`

**Authentication:** Required

**Query Parameters:**
- `status_filter` (optional): Filter by status
- `symbol` (optional): Filter by stock symbol
- `limit` (optional, default: 50, max: 100)
- `offset` (optional, default: 0)

**Response:** `200 OK`
```json
{
  "orders": [
    {
      "id": "uuid",
      "user_id": "uuid",
      "symbol": "AAPL",
      "order_type": "LIMIT",
      "side": "BUY",
      "quantity": 10,
      "price": 175.00,
      "status": "FILLED",
      "filled_quantity": 10,
      "average_price": 175.50,
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 1
}
```

### Cancel Order

Cancel a pending order.

**Endpoint:** `DELETE /api/v1/trading/orders/{order_id}`

**Authentication:** Required

**Response:** `204 No Content`

### Get Portfolio

Get current portfolio with all holdings.

**Endpoint:** `GET /api/v1/trading/portfolio`

**Authentication:** Required

**Response:** `200 OK`
```json
{
  "holdings": [
    {
      "symbol": "AAPL",
      "quantity": 10,
      "average_cost": 175.00,
      "current_price": 175.50,
      "total_value": 1755.00,
      "unrealized_pnl": 5.00
    }
  ],
  "total_value": 1755.00,
  "cash_balance": 10000.00
}
```

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Specific error message",
  "status_code": 400
}
```

### Common Errors

**Validation Error (422)**
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "value is not a valid email address",
      "type": "value_error.email"
    }
  ]
}
```

**Unauthorized (401)**
```json
{
  "detail": "Invalid authentication credentials"
}
```

**Forbidden (403)**
```json
{
  "detail": "Not enough permissions"
}
```

**Not Found (404)**
```json
{
  "detail": "Resource not found"
}
```

## Pagination

List endpoints support pagination via query parameters:

- `limit`: Number of items to return (default: 50, max: 100)
- `offset`: Number of items to skip (default: 0)

Example:
```
GET /api/v1/banking/transactions/history?account_id=uuid&limit=20&offset=40
```

## Filtering

Many endpoints support filtering via query parameters. Check individual endpoint documentation for available filters.

## WebSocket Support (Coming Soon)

Real-time market data will be available via WebSocket connections at:

```
ws://localhost:8000/ws/market-data
```

## SDK and Client Libraries

Official client libraries:
- Python SDK (coming soon)
- JavaScript/TypeScript SDK (coming soon)
- Mobile SDKs (iOS/Android) (coming soon)
