# NASDAQ Stock Market Simulator - Phase 1: Foundation

## Overview
This is Phase 1 of the NASDAQ Stock Market Simulator project, focusing on establishing the foundational components of the Order Management System (OMS).

## Phase 1 Objectives (Weeks 1-4)

### Infrastructure Setup
- ✅ Basic development environment
- ✅ Core OMS service structure
- ✅ REST API endpoints
- ✅ Basic data models

### Core OMS Development
- ✅ Order data models and validation
- ✅ Basic order lifecycle management
- ✅ In-memory data storage
- ✅ REST API endpoints

## Features
- **Order Management**: Submit, track, and cancel orders
- **REST API**: HTTP endpoints for order operations
- **Data Models**: Basic order, position, and trade models
- **Validation**: Input validation and error handling

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python main.py
```

The application will start on `http://localhost:8000`

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### Example Usage

#### Submit an Order
```bash
curl -X POST "http://localhost:8000/api/v1/orders/" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "AAPL",
       "side": "BUY",
       "order_type": "LIMIT",
       "quantity": 100,
       "price": 150.00
     }'
```

#### Get All Orders
```bash
curl "http://localhost:8000/api/v1/orders/"
```

#### Get Specific Order
```bash
curl "http://localhost:8000/api/v1/orders/{order_id}"
```

#### Cancel Order
```bash
curl -X DELETE "http://localhost:8000/api/v1/orders/{order_id}"
```

## Architecture
```
Phase 1 Architecture:
├── Core Models (Order, Position, Trade)
├── OMS Service (Order Management)
├── Repository Layer (In-memory storage)
├── REST API Layer (FastAPI)
└── Basic Validation
```

## Next Steps
After completing Phase 1, proceed to:
- **Phase 2**: Protocol Integration (FIX, FAST, ITCH)
- **Phase 3**: Market Data and Risk Management
- **Phase 4**: Advanced Features and Analytics

## Development Guidelines
1. All code should include proper error handling
2. Use async/await for all operations
3. Include unit tests for all business logic
4. Follow Python PEP 8 style guidelines
5. Document all public APIs

## Testing
```bash
# Run unit tests (when implemented)
python -m pytest tests/unit/

# Run integration tests (when implemented)  
python -m pytest tests/integration/
```

## Contributing
1. Follow the established code structure
2. Add tests for new functionality
3. Update documentation for any API changes
4. Ensure all changes are backward compatible
