# NASDAQ Stock Market Simulator - Phase 2: Protocol Integration

## Overview
This is Phase 2 of the NASDAQ Stock Market Simulator project, focusing on integrating FIX, FAST, and ITCH protocol gateways with the Order Management System (OMS).

## Phase 2 Objectives (Weeks 5-8)

### FIX Gateway Enhancement
- ✅ Enhanced FIX protocol server with session management
- ✅ Order routing to OMS
- ✅ Error handling and recovery
- ✅ Support for FIX 4.4 messages

## Features
- **FIX Protocol Support**: FIX 4.4 message handling
- **Session Management**: FIX session handling with heartbeats
- **Protocol Integration**: Seamless routing to OMS

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python main.py
```

The application starts:
- REST API: `http://localhost:8000`
- FIX Gateway: `localhost:9878`

### Protocol Testing

#### FIX Protocol Testing
```bash
# Connect to FIX server using telnet or FIX client
telnet localhost 9878

# Send FIX Logon message (SOH replaced with |)
8=FIX.4.4|9=65|35=A|34=1|49=CLIENT|56=NASDAQ_SIM|52=20241201-12:00:00|98=0|108=30|10=000|

# Send New Order Single
8=FIX.4.4|9=120|35=D|34=2|49=CLIENT|56=NASDAQ_SIM|52=20241201-12:00:01|11=ORDER1|55=AAPL|54=1|38=100|40=2|44=150.00|10=000|
```

## Architecture
```
Phase 2 Architecture:
├── FIX Gateway (Port 9878)
│   ├── Session Management
│   ├── Message Parsing
│   └── Order Routing
└── Enhanced OMS
    ├── Multi-protocol Support
    ├── Message Validation
    └── Response Generation
```

## Protocol Messages

### FIX Messages Supported
- **Logon (A)**: Session establishment
- **Heartbeat (0)**: Keep-alive messages
- **New Order Single (D)**: Order submission
- **Order Cancel Request (F)**: Order cancellation
- **Execution Report (8)**: Order status updates

## Next Steps
After completing Phase 2, proceed to:
- **Phase 3**: Market Data and Risk Management
- **Phase 4**: Advanced Features and Analytics
