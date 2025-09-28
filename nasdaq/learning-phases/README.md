# NASDAQ Stock Market Simulator - Phased Development

> 📚 **Learning Path**: This directory contains the **step-by-step, educational** implementation of the NASDAQ Stock Market Simulator project. For the complete, production-ready version, see [`../prod-app/`](../prod-app/).

This directory guides you through building the NASDAQ simulator incrementally across 4 phases, perfect for understanding the architecture and learning each component.

## Phase Structure

### Phase 1: Foundation (Weeks 1-4) 
**Directory**: `phase_1/`
**Focus**: Core OMS development and basic infrastructure

**Key Components**:
- Basic Order Management System (OMS)
- Order data models and validation
- REST API endpoints
- In-memory data storage
- Development environment setup

**Run Phase 1**:
```bash
cd phase_1
pip install -r requirements.txt  
python main.py
```

### Phase 2: Protocol Integration (Weeks 5-8)
**Directory**: `phase_2/`
**Focus**: FIX, FAST, and ITCH protocol gateways

**Key Components**:
- Enhanced FIX protocol server (Port 9878)
- Protocol message routing to OMS
- Session management and error handling

**Run Phase 2**:
```bash
cd phase_2
pip install -r requirements.txt
python main.py
```

### Phase 3: Market Data and Risk (Weeks 9-12)
**Directory**: `phase_3/` (Coming Soon)
**Focus**: Real-time market data and risk management

### Phase 4: Advanced Features (Weeks 13-16)  
**Directory**: `phase_4/` (Coming Soon)
**Focus**: Settlement, analytics, and production readiness

## Development Workflow

1. **Start with Phase 1**: Build foundation components
2. **Progress to Phase 2**: Add protocol support
3. **Continue to Phase 3**: Add market data and risk
4. **Complete with Phase 4**: Add advanced features

## Testing Each Phase

Each phase includes:
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- README with specific instructions
- Requirements file
- Example usage

## Architecture Evolution

```
Phase 1: OMS Core → Phase 2: Protocol Gateways → Phase 3: Market Data & Risk → Phase 4: Advanced Features
```

## Contributing

1. Work on one phase at a time
2. Complete all features in current phase before moving to next
3. Maintain backward compatibility between phases
4. Follow the established code structure
5. Add comprehensive tests

## Support

Each phase directory contains:
- Detailed README with setup instructions
- Code examples and usage
- Architecture documentation
- Testing guidelines
