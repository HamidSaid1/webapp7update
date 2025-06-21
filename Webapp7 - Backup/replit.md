# AutoTrade Plus - Sophisticated Stock Trading Platform

## Overview

AutoTrade Plus is a sophisticated Python-based stock trading platform that provides real-time market analysis, automated trading strategies, and comprehensive risk management. The application features a Flask web interface with real-time data visualization and integrates with Interactive Brokers (IBKR) for market data and trade execution.

## System Architecture

The application follows a modular, event-driven architecture with clear separation of concerns:

- **Frontend**: Flask web application with responsive Bootstrap UI and real-time JavaScript updates
- **Backend**: Asynchronous trading engine with IBKR API integration using ib-insync
- **Analysis Engine**: Advanced technical analysis module with multiple indicators and adaptive algorithms
- **Risk Management**: Comprehensive trade validation and position management
- **Configuration**: Centralized configuration management with dataclass-based parameters
- **Deployment**: Gunicorn WSGI server with autoscale deployment strategy

## Key Components

### 1. Trading Engine (`trading_engine.py`)
- **Purpose**: Core trading logic and market data processing
- **Architecture**: Event-driven design with asyncio for concurrent operations
- **Key Features**:
  - Real-time market data simulation and processing
  - Advanced technical analysis calculations (RSI, Bollinger Bands, Stochastic)
  - Position tracking and P&L calculation
  - Trade execution with stop-loss mechanisms
  - Force trade mode with cooldown periods

### 2. Configuration Management (`config.py`)
- **Purpose**: Centralized parameter management using dataclasses
- **Architecture**: Type-safe configuration with environment variable support
- **Parameters Include**:
  - IBKR connection settings (host, port, client ID)
  - Trading parameters (equity per trade, stop loss, max trades)
  - Technical analysis settings (RSI period, Bollinger periods, Stochastic settings)
  - Data collection parameters (chart intervals, data points)

### 3. Web Interface (`app.py`)
- **Purpose**: Flask-based web application for user interaction
- **Architecture**: RESTful API design with template-based rendering
- **Features**:
  - Real-time dashboard with market data display
  - Settings configuration interface
  - Analytics and performance tracking
  - Responsive Bootstrap-based UI

### 4. Frontend Components
- **Dashboard**: Real-time market data table with technical indicators
- **Settings**: Parameter configuration matching reserve_file.py specifications
- **Analytics**: Performance tracking and trade history visualization
- **JavaScript**: Real-time updates using fetch API with 5-second intervals

## Data Flow

1. **Market Data Collection**: Trading engine collects and processes market data
2. **Technical Analysis**: Real-time calculation of RSI, Bollinger Bands, and Stochastic indicators
3. **Signal Generation**: Analysis of price movements and technical indicator changes
4. **Trade Decision**: Evaluation of trading criteria and risk parameters
5. **Web Updates**: Real-time data pushed to frontend via API endpoints
6. **User Interaction**: Configuration changes and manual controls through web interface

## External Dependencies

### Core Dependencies
- **Flask**: Web framework for the application interface
- **ib-insync**: Interactive Brokers API integration
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations for technical analysis
- **scikit-learn**: Linear regression for slope calculations
- **nest-asyncio**: Asyncio compatibility for nested event loops

### Frontend Dependencies
- **Bootstrap 5.3.0**: Responsive CSS framework
- **Feather Icons**: Icon library for UI elements
- **JavaScript Fetch API**: Real-time data updates

### Database Dependencies
- **psycopg2-binary**: PostgreSQL adapter (prepared for future database integration)
- **flask-sqlalchemy**: SQL toolkit (prepared for data persistence)

## Deployment Strategy

### Production Deployment
- **WSGI Server**: Gunicorn with bind to 0.0.0.0:5000
- **Deployment Target**: Autoscale configuration for dynamic scaling
- **Environment**: Python 3.11 with Nix package management
- **Process Management**: Replit workflows with parallel task execution

### Development Environment
- **Hot Reload**: Gunicorn with --reload flag for development
- **Port Configuration**: Configurable port with waitForPort health checks
- **Asset Management**: Static file serving for CSS/JS resources

### System Requirements
- **Python**: Version 3.11 or higher
- **PostgreSQL**: Database server (configured but not yet implemented)
- **OpenSSL**: Secure connections for IBKR API
- **Memory**: Sufficient for real-time data processing and web serving

## Changelog

- June 20, 2025: Initial setup with Flask web application
- June 20, 2025: Added TWS paper trading connectivity with real IBKR API integration
- June 20, 2025: Fixed JSON serialization errors and implemented dual-mode operation (real TWS + simulation fallback)

## User Preferences

Preferred communication style: Simple, everyday language.