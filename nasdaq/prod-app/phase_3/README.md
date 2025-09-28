# NASDAQ Stock Market Simulator - Phase 3: Market Data and Risk

## Overview
This is Phase 3 of the NASDAQ Stock Market Simulator project, focusing on real-time market data distribution and comprehensive risk management.

## Phase 3 Objectives (Weeks 9-12)

### Market Data Engine
- ✅ Real-time data distribution
- ✅ Market data simulation
- ✅ Trade history tracking
- ✅ Subscription management

### Risk Management
- ✅ Pre-trade risk checks
- ✅ Position tracking
- ✅ Risk limit monitoring
- ✅ Alert generation

## Features
- **Real-time Market Data**: Live price feeds and trade data
- **Risk Management**: Pre-trade checks and position monitoring
- **Trade Simulation**: Realistic market behavior simulation
- **Risk Alerts**: Automated risk monitoring and alerting
- **Position Tracking**: Real-time position management

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python main.py
```

The application starts on `http://localhost:8000`

### API Endpoints

#### Market Data
```bash
# Get all market data
curl "http://localhost:8000/market-data"

# Get specific symbol data
curl "http://localhost:8000/market-data/AAPL"

# Get trade history
curl "http://localhost:8000/trades?limit=50"

# Get trades for specific symbol
curl "http://localhost:8000/trades?symbol=AAPL&limit=20"
```

#### Risk Management
```bash
# Get risk summary
curl "http://localhost:8000/risk/summary"
```

## Architecture
```
Phase 3 Architecture:
├── Market Data Engine
│   ├── Real-time Price Simulation
│   ├── Trade Generation
│   ├── Subscription Management
│   └── Data Distribution
├── Risk Management Engine
│   ├── Pre-trade Checks
│   ├── Position Tracking
│   ├── Limit Monitoring
│   └── Alert Generation
└── Enhanced OMS
    ├── Risk Integration
    ├── Market Data Integration
    └── Real-time Processing
```

## Market Data Features
- **Real-time Updates**: Market data updates every second
- **Multiple Symbols**: AAPL, GOOGL, MSFT, TSLA, NVDA, AMZN, META
- **Bid/Ask Spreads**: Realistic bid-ask spread simulation
- **Trade Generation**: Random trade execution simulation
- **Historical Data**: Trade history storage

## Risk Management Features
- **Order Size Limits**: Maximum quantity and value per order
- **Position Limits**: Maximum position size per symbol
- **Risk Levels**: LOW, MEDIUM, HIGH, CRITICAL risk classification
- **Real-time Monitoring**: Continuous risk assessment
- **Alert System**: Automated risk alert generation

## Next Steps
After completing Phase 3, proceed to:
- **Phase 4**: Advanced Features and Analytics

## Testing
```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests  
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/
```
