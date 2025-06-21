# TWS Paper Trading Setup Guide

## Prerequisites
1. **IBKR Account** with paper trading enabled
2. **TWS (Trader Workstation)** or **IB Gateway** installed
3. **API permissions** enabled in your IBKR account

## Step 1: Enable API in TWS
1. Open TWS and log into your paper trading account
2. Go to **File** → **Global Configuration**
3. Navigate to **API** → **Settings**
4. Check "Enable ActiveX and Socket Clients"
5. Set **Socket port**: `7497` (paper trading)
6. Uncheck "Read-Only API" if you want to place trades
7. Click **OK** and restart TWS

## Step 2: Configure Paper Trading Connection
The app is already configured for paper trading:
- **Host**: 127.0.0.1 (localhost)
- **Port**: 7497 (paper trading port)
- **Client ID**: 2
- **Paper Trading**: Enabled

## Step 3: Test Connection
1. Make sure TWS is running and logged into paper trading
2. Click "Connect" in the AutoTrade Plus interface
3. Look for "Connected to IBKR TWS (Paper Trading)" message
4. If connection fails, the app falls back to simulation mode

## Step 4: Start Trading
1. Click "Start Stream" to begin market data
2. The system will use real IBKR market data if connected
3. Trades will be executed as paper trades through TWS
4. Monitor trades in both AutoTrade Plus and TWS

## Troubleshooting
- **Connection Failed**: Ensure TWS is running with API enabled
- **No Market Data**: Check your IBKR market data subscriptions
- **Orders Not Filling**: Verify paper trading account has sufficient funds
- **Port Issues**: Make sure port 7497 is not blocked by firewall

## Trading Features
- **Real-time market data** from IBKR
- **Authentic calculations** from reserve_file.py
- **Paper trade execution** through TWS API
- **Risk management** with stop-loss and position sizing
- **Performance tracking** and analytics

The system seamlessly switches between real TWS connectivity and simulation mode based on availability.