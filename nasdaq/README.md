# NASDAQ Stock Market Simulator

This directory contains two complementary implementations of the NASDAQ Stock Market Simulator project, designed to serve different learning and deployment purposes.

## Directory Structure

### 📚 `learning-phases/` - Educational Development Path
**Purpose**: Step-by-step, phased learning approach  
**Audience**: Students, developers learning the system architecture  
**Approach**: Incremental development across 4 distinct phases

This directory guides you through building the NASDAQ simulator from scratch:
- **Phase 1**: Foundation (OMS core, basic REST API)
- **Phase 2**: Protocol Integration (FIX, FAST, ITCH gateways)  
- **Phase 3**: Market Data & Risk Management
- **Phase 4**: Advanced Features & Production Readiness

Each phase builds upon the previous, allowing you to understand the architecture evolution and master each component before proceeding.

### 🚀 `prod-app/` - Complete Integrated Application
**Purpose**: Production-ready, fully integrated implementation  
**Audience**: Deployment teams, system administrators, production users  
**Approach**: Complete system with all features integrated

This directory contains the finished, production-ready application with all phases combined into a cohesive, optimized system ready for deployment.

## Getting Started

### For Learning (Recommended for New Users)
```bash
cd learning-phases/
# Start with Phase 1 and progress sequentially
cd phase_1/
pip install -r requirements.txt
python main.py
```

### For Production Deployment
```bash
cd prod-app/
# Complete system ready to run
cd phase_4/  # or whichever phase represents the full system
pip install -r requirements.txt
python main.py
```

## Navigation Guide

- **New to the project?** → Start with `learning-phases/`
- **Ready for deployment?** → Use `prod-app/`
- **Want to understand architecture?** → Study `learning-phases/` progression
- **Need production features?** → Deploy from `prod-app/`

Both directories maintain their own detailed README files with specific setup and usage instructions.