# IB Gateway API Setup Instructions

## Current Issue
The system is trying all common ports (4001, 4002, 7497, 7496) but getting "Connection Refused" errors. This means the API is not enabled in your IB Gateway.

## Fix Steps for IB Gateway:

### 1. Enable API in IB Gateway
1. **Open IB Gateway** (not TWS)
2. **Login** to your paper trading account
3. Click **Configure** button
4. Go to **API** tab
5. Check **"Enable ActiveX and Socket Clients"**
6. Set **Socket port** to: `4001` (for paper trading)
7. **Uncheck** "Read-Only API" (to allow trading)
8. Set **Master API client ID** to: `0`
9. Click **OK**

### 2. Restart IB Gateway
- Close IB Gateway completely
- Restart IB Gateway
- Login again

### 3. Test Connection
- Click "Connect" in AutoTrade Plus
- Should see "Connected to IB Gateway Paper Trading"

## Alternative: Use TWS Instead
If you prefer TWS over Gateway:
1. Open TWS
2. File → Global Configuration
3. API → Settings
4. Enable ActiveX and Socket Clients
5. Socket port: 7497
6. Restart TWS

## Current Status
✓ AutoTrade Plus is fully functional with simulation
✓ All exact reserve_file.py calculations preserved
✓ Ready to connect to real IBKR data once API is enabled
✓ Form validation and trading logic working perfectly

The preview has complete functionality - it just needs the IBKR API enabled to get real market data instead of simulated quotes.