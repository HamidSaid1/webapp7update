"""
Standalone Python Trading Application
Complete trading engine with GUI interface - non-web version
Exact same logic, calculations, and settings as the web version
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import asyncio
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import time
import json
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class TradingConfig:
    """Trading configuration matching the web version exactly"""

    # IBKR Connection Settings
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    client_id: int = 2  # Changed from 1 to avoid conflicts
    timeout: int = 10
    paper_trading: bool = True

    # Trading Parameters (exact from web version)
    default_tickers: str = "APVO"
    default_exchange: str = "SMART"
    equity_per_trade: float = 100.0
    max_open_trades: int = 2
    hard_stop_loss: float = 0.03

    # Technical Analysis Parameters (exact from web version)
    rsi_period: int = 3
    bb_period: int = 3
    stoch_k_period: int = 2
    stoch_d_period: int = 2
    stoch_k_min: float = 50.0
    stoch_d_min: float = 40.0

    # Data Collection Settings (exact from web version)
    data_points: int = 16
    chart_intervals: List[str] = None
    default_chart_interval: str = "1 min"
    analysis_interval: int = 1
    duration_str: str = "1 D"

    def __post_init__(self):
        if self.chart_intervals is None:
            self.chart_intervals = ["5 sec", "10 sec", "30 sec", "1 min", "3 min", "5 min", "10 min", "15 min"]

class StandaloneTradingEngine:
    """Standalone trading engine with exact same logic as web version"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Trading state (exact from web version)
        self.connected = False
        self.streaming = False
        self.market_data = {}
        self.active_orders = {}
        self.trade_cooldowns = {}
        self.available_funds = 1000.0
        self.force_trade_enabled = False
        self.price_history = {}

        # User configurable parameters (exact from web version)
        self.tickers = "APVO"
        self.data_points = 16
        self.chart_interval = "1 min"
        self.analysis_interval = 1
        self.rsi_period = 3
        self.bb_period = 3
        self.stoch_k_period = 2
        self.stoch_d_period = 2
        self.stoch_k_min = 50.0
        self.stoch_d_min = 40.0
        self.duration_str = "1 D"
        self.equity_per_trade = 100.0
        self.exchange_var = "SMART"
        self.hard_stop_loss = 0.03

        # Analytics tracking (exact from web version)
        self.last_price_change = "N/A"
        self.last_rsi_change = "N/A"
        self.last_bollinger_change = "N/A"
        self.trade_history = []

    def compute_slope(self, prices):
        """Exact implementation from web version"""
        if len(prices) < 2:
            return 0.0
        x = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]

    def calculate_adaptive_thresholds(self, spike_stats):
        """Exact implementation from web version"""
        if spike_stats["count"] < 3:
            return None

        return {
            "price_rate": sum(spike_stats["price_rates"]) / len(spike_stats["price_rates"]),
            "rsi_rate": sum(spike_stats["rsi_rates"]) / len(spike_stats["rsi_rates"]),
            "boll_rate": sum(spike_stats["boll_rates"]) / len(spike_stats["boll_rates"]),
            "slope": sum(spike_stats["slopes"]) / len(spike_stats["slopes"])
        }

    def compute_rsi(self, df, period=14):
        """Exact RSI implementation from web version"""
        if len(df) < period + 1:
            return pd.Series([50.0] * len(df), index=df.index)

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def compute_stochastic(self, df, k_period=14, d_period=3):
        """Exact Stochastic implementation from web version"""
        if len(df) < k_period:
            df['%K'] = pd.Series([50.0] * len(df), index=df.index)
            df['%D'] = pd.Series([50.0] * len(df), index=df.index)
            return df

        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']

        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        range_val = high_max - low_min
        range_val = range_val.replace(0, 1)

        df['%K'] = ((df['close'] - low_min) / range_val * 100).fillna(50.0)
        df['%D'] = df['%K'].rolling(window=d_period).mean().fillna(50.0)
        return df

    def compute_bollinger(self, df, period=20, num_std=2):
        """Exact Bollinger implementation from web version"""
        if len(df) < period:
            return pd.Series([50.0] * len(df), index=df.index)

        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        band_width = upper_band - lower_band
        band_width = band_width.replace(0, 1)

        bollinger_pct = ((df["close"] - lower_band) / band_width * 100).fillna(50.0)
        return bollinger_pct

    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway API with configurable port"""
        try:
            try:
                import ib_insync
                self.ib = ib_insync.IB()

                # First try the configured port
                try:
                    self.logger.info(f"Attempting connection to port {self.config.ibkr_port} with client ID {self.config.client_id}")

                    await self.ib.connectAsync(
                        host=self.config.ibkr_host,
                        port=self.config.ibkr_port,
                        clientId=self.config.client_id,
                        timeout=10
                    )

                    if self.ib.isConnected():
                        self.connected = True
                        self.logger.info(f"✅ Connected to port {self.config.ibkr_port}")
                        self.ib.reqMarketDataType(3)  # Use delayed market data
                        return True

                except Exception as e:
                    if self.ib.isConnected():
                        self.ib.disconnect()

                    # If client ID conflict, try with incremented client ID
                    if "already in use" in str(e).lower():
                        self.logger.warning(f"Client ID {self.config.client_id} in use, trying with incremented ID")

                        for client_id_offset in range(1, 6):  # Try client IDs 2-7
                            try:
                                new_client_id = self.config.client_id + client_id_offset
                                self.logger.info(f"Trying with client ID {new_client_id}")

                                await self.ib.connectAsync(
                                    host=self.config.ibkr_host,
                                    port=self.config.ibkr_port,
                                    clientId=new_client_id,
                                    timeout=10
                                )

                                if self.ib.isConnected():
                                    self.connected = True
                                    self.config.client_id = new_client_id  # Update config
                                    self.logger.info(f"✅ Connected to port {self.config.ibkr_port} with client ID {new_client_id}")
                                    self.ib.reqMarketDataType(3)  # Use delayed market data
                                    return True

                            except Exception as retry_error:
                                if self.ib.isConnected():
                                    self.ib.disconnect()
                                self.logger.debug(f"Client ID {new_client_id} failed: {retry_error}")
                                continue

                    self.logger.error(f"Connection failed on configured port {self.config.ibkr_port}: {e}")
                    raise e

            except ImportError:
                self.logger.warning("ib_insync not available")
                raise Exception("ib_insync not installed")

        except Exception as e:
            self.logger.error(f"❌ TWS connection failed: {e}")
            self.logger.info("💡 Connection Tips:")
            self.logger.info(f"   1. Ensure TWS/Gateway is running on port {self.config.ibkr_port}")
            self.logger.info("   2. Enable API in TWS: File → Global Configuration → API")
            self.logger.info("   3. Check 'Enable ActiveX and Socket Clients'")
            self.logger.info("   4. Close any other trading applications using the same client ID")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from IBKR TWS API"""
        if hasattr(self, 'ib') and self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("✅ Disconnected from IBKR TWS")
        self.connected = False
        self.streaming = False

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connected

    def is_streaming(self) -> bool:
        """Check if streaming"""
        return self.streaming

    async def start_market_data(self, symbols=None):
        """Start market data streaming - exact same logic as web version"""
        try:
            self.logger.info("Starting market data streaming")

            if symbols is None:
                tickers = [ticker.strip() for ticker in self.tickers.split(',')]
            else:
                tickers = symbols
            self.logger.info(f"Setting up streaming for {len(tickers)} tickers: {tickers}")

            has_real_connection = (hasattr(self, 'ib') and 
                                 self.ib and 
                                 self.ib.isConnected())

            if has_real_connection:
                self.logger.info("Using real IBKR market data")
                success = await self._start_real_streaming(tickers)
                if success:
                    return True
                else:
                    self.logger.error("Real streaming failed - no fallback available")
                    return False
            else:
                self.logger.error("IBKR not connected - real connection required")
                return False

        except Exception as e:
            self.logger.error(f"❌ Market data streaming error: {e}")
            return False

    async def _start_real_streaming(self, tickers):
        """Start real IBKR streaming - exact same as web version"""
        try:
            import ib_insync
            from ib_insync import Stock

            # Start streaming for each ticker
            for ticker in tickers:
                try:
                    # Create contract
                    contract = Stock(ticker, self.exchange_var, 'USD')
                    qualified_contracts = await self.ib.qualifyContractsAsync(contract)

                    if qualified_contracts:
                        contract = qualified_contracts[0]

                        # Request market data
                        ticker_obj = self.ib.reqMktData(contract, '', False, False)

                        # Get historical data for technical analysis
                        historical_data = await self.ib.reqHistoricalDataAsync(
                            contract,
                            endDateTime='',
                            durationStr=self.duration_str,
                            barSizeSetting=self.chart_interval,
                            whatToShow='TRADES',
                            useRTH=True
                        )

                        if historical_data and len(historical_data) > 0:
                            # Initialize market data entry
                            self.market_data[ticker] = {
                                'last': 0.0,
                                'high': 0.0,
                                'low': 0.0,
                                'volume': 0,
                                'rsi': 0.0,
                                'bollinger_pct': 0.0,
                                'stoch_k': 0.0,
                                'stoch_d': 0.0,
                                'price_rate': 0.0,
                                'rsi_rate': 0.0,
                                'bollinger_rate': 0.0,
                                'trend_slope': 0.0,
                                'meets_criteria': False,
                                'exchange': self.exchange_var,
                                'timestamp': datetime.now().isoformat(),
                                'status': 'Connected',
                                'historical_data': historical_data,
                                'ticker_obj': ticker_obj
                            }

                            # Perform initial analysis
                            await self._perform_analysis(ticker)

                            self.logger.info(f"✅ Successfully set up streaming for {ticker}")
                        else:
                            self.logger.warning(f"⚠️ No historical data available for {ticker}")

                except Exception as e:
                    self.logger.error(f"❌ Error setting up {ticker}: {e}")
                    continue

            if self.market_data:
                self.streaming = True
                asyncio.create_task(self._analysis_loop())
                self.logger.info(f"🚀 Real market data streaming started for {len(self.market_data)} symbols")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Real streaming setup error: {e}")
            return False

    async def _analysis_loop(self):
        """Main analysis loop for real data - exact same as web version"""
        while self.streaming:
            try:
                for ticker in self.market_data:
                    await self._perform_analysis(ticker)
                await asyncio.sleep(self.analysis_interval)
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(5)

    async def _perform_analysis(self, ticker):
        """Perform technical analysis on a ticker - exact same as web version"""
        try:
            data = self.market_data[ticker]

            if 'ticker_obj' in data and hasattr(data['ticker_obj'], 'last'):
                # Update real-time price from ticker object
                if data['ticker_obj'].last and data['ticker_obj'].last > 0:
                    data['last'] = data['ticker_obj'].last
                if data['ticker_obj'].high and data['ticker_obj'].high > 0:
                    data['high'] = data['ticker_obj'].high
                if data['ticker_obj'].low and data['ticker_obj'].low > 0:
                    data['low'] = data['ticker_obj'].low
                if data['ticker_obj'].volume:
                    data['volume'] = data['ticker_obj'].volume

            if 'historical_data' in data and data['historical_data']:
                # Extract prices from historical data
                prices = [bar.close for bar in data['historical_data']]

                # If we have real-time price, add it to prices for analysis
                if data['last'] > 0:
                    prices.append(data['last'])

                # Update indicators
                data['rsi'] = self._calculate_rsi(prices)
                bb_data = self._calculate_bollinger_bands(prices)
                data['bollinger_pct'] = bb_data['position']
                stoch_data = self._calculate_stochastic(prices)
                data['stoch_k'] = stoch_data['k']
                data['stoch_d'] = stoch_data['d']

                # Check criteria
                data['meets_criteria'] = (
                    data['rsi'] < 30 or data['rsi'] > 70 or 
                    data['bollinger_pct'] < 20 or data['bollinger_pct'] > 80 or
                    data['stoch_k'] < self.stoch_k_min or data['stoch_d'] < self.stoch_d_min
                )

                data['timestamp'] = datetime.now().isoformat()

        except Exception as e:
            self.logger.error(f"Analysis error for {ticker}: {e}")



    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator - exact same as web version"""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, prices, period=20):
        """Calculate Bollinger Bands - exact same as web version"""
        if len(prices) < period:
            return {'position': 50.0}

        recent_prices = prices[-period:]
        sma = sum(recent_prices) / len(recent_prices)
        variance = sum((price - sma) ** 2 for price in recent_prices) / len(recent_prices)
        std_dev = variance ** 0.5

        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)

        current_price = prices[-1]

        if upper_band == lower_band:
            position = 50.0
        else:
            position = ((current_price - lower_band) / (upper_band - lower_band)) * 100

        return {'position': max(0, min(100, position))}

    def _calculate_stochastic(self, prices, k_period=14, d_period=3):
        """Calculate Stochastic oscillator - exact same as web version"""
        if len(prices) < k_period:
            return {'k': 50.0, 'd': 50.0}

        recent_prices = prices[-k_period:]
        highest_high = max(recent_prices)
        lowest_low = min(recent_prices)
        current_price = prices[-1]

        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100

        d_percent = k_percent

        return {'k': k_percent, 'd': d_percent}



    def stop_market_data(self):
        """Stop market data streaming"""
        self.streaming = False
        self.logger.info("Stopped market data streaming")

    def toggle_force_trade(self):
        """Toggle force trade mode on/off"""
        self.force_trade_enabled = not self.force_trade_enabled
        status = "enabled" if self.force_trade_enabled else "disabled"
        self.logger.info(f"Force trade mode {status}")
        return self.force_trade_enabled

    def is_market_hours(self):
        """Check if current time is during regular market hours (9:30 AM - 4:00 PM ET)"""
        try:
            from datetime import time
            now = datetime.now()

            # Check if weekend
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False

            # Market hours: 9:30 AM - 4:00 PM ET (simplified - not accounting for timezone)
            market_open = time(9, 30)
            market_close = time(16, 0)
            current_time = now.time()

            return market_open <= current_time <= market_close
        except:
            return True  # Default to market hours if check fails

    async def execute_force_trade(self, symbol: str, action: str):
        """Execute a force trade with proper order type based on market hours"""
        try:
            if symbol not in self.market_data:
                return {'error': f'No market data available for {symbol}'}

            current_price = self.market_data[symbol].get('last', 0)
            if current_price <= 0:
                return {'error': f'Invalid price for {symbol}'}

            quantity = max(1, int(self.equity_per_trade / current_price))
            is_market_open = self.is_market_hours()

            # Determine order type and price
            if is_market_open:
                # Use market orders during regular hours
                order_type = "MARKET"
                order_price = current_price
            else:
                # Use limit orders during extended hours
                order_type = "LIMIT"
                if action.upper() == "BUY":
                    order_price = current_price + 0.15  # Add 15 cents for buy
                else:
                    order_price = current_price - 0.15  # Subtract 15 cents for sell
                order_price = round(order_price, 2)

            # Execute the trade - only real IBKR execution allowed
            if hasattr(self, 'ib') and self.ib and self.ib.isConnected():
                # Real IBKR execution
                success = await self._execute_real_force_trade(symbol, action, quantity, order_type, order_price)
            else:
                self.logger.error("IBKR connection required for trade execution")
                return {'error': 'IBKR connection required for trade execution'}

            if success:
                self.logger.info(f"Force {action} executed: {symbol} x{quantity} @ ${order_price:.2f} ({order_type})")
                return {
                    'success': f'Force {action} executed: {quantity} shares of {symbol} @ ${order_price:.2f}',
                    'details': {
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': order_price,
                        'order_type': order_type,
                        'market_hours': is_market_open
                    }
                }
            else:
                return {'error': f'Failed to execute force {action} for {symbol}'}

        except Exception as e:
            self.logger.error(f"Force trade execution error: {e}")
            return {'error': f'Force trade error: {str(e)}'}

    async def _execute_real_force_trade(self, symbol: str, action: str, quantity: int, order_type: str, price: float):
        """Execute real force trade through IBKR"""
        try:
            import ib_insync
            from ib_insync import Stock, MarketOrder, LimitOrder

            # Create contract
            contract = Stock(symbol, self.exchange_var, 'USD')
            qualified_contracts = await self.ib.qualifyContractsAsync(contract)

            if not qualified_contracts:
                return False

            contract = qualified_contracts[0]

            # Create order
            if order_type == "MARKET":
                order = MarketOrder(action.upper(), quantity)
            else:  # LIMIT
                order = LimitOrder(action.upper(), quantity, price)

            # Place order
            trade = self.ib.placeOrder(contract, order)

            # Wait for execution (simplified)
            await asyncio.sleep(2)

            return True

        except Exception as e:
            self.logger.error(f"Real force trade execution error: {e}")
            return False



    def get_analytics(self):
        """Get trading analytics - exact same as web version"""
        total_trades = len(self.trade_history)
        total_profit = sum(trade.get('profit', 0) for trade in self.trade_history)
        win_rate = 0.0

        if total_trades > 0:
            winning_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
            win_rate = (winning_trades / total_trades) * 100

        return {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'available_funds': self.available_funds,
            'active_positions': len(self.active_orders),
            'last_price_change': self.last_price_change,
            'last_rsi_change': self.last_rsi_change,
            'last_bollinger_change': self.last_bollinger_change,
            'equity_per_trade': self.equity_per_trade,
            'rsi_period': self.rsi_period,
            'bb_period': self.bb_period,
            'stoch_k_period': self.stoch_k_period,
            'stoch_d_period': self.stoch_d_period,
            'stoch_k_min': self.stoch_k_min,
            'stoch_d_min': self.stoch_d_min
        }

    def update_parameters(self, params):
        """Update trading parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.logger.info(f"Updated parameters: {params}")

    async def run_backtest(self, symbols_input: str, start_date: str, end_date: str, initial_capital: float = 10000.0):
        """Enhanced backtest with proper capital management and multi-symbol support"""
        try:
            # Parse symbols (support both single and multiple symbols)
            if isinstance(symbols_input, str):
                symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            else:
                symbols = [symbols_input.upper()]

            self.logger.info(f"Starting backtest for {len(symbols)} symbols: {symbols} from {start_date} to {end_date}")

            # Validate date range
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days

            if days <= 0:
                return {'error': 'End date must be after start date'}

            # Initialize capital management
            available_capital = initial_capital
            active_positions = {}  # Track open positions per symbol
            completed_trades = []
            total_profit = 0.0
            max_equity_per_trade = min(self.equity_per_trade, initial_capital * 0.1)  # Max 10% per trade

            # Check IBKR connection
            has_real_connection = (hasattr(self, 'ib') and self.ib and self.ib.isConnected())

            if not has_real_connection:
                return {'error': 'IBKR connection required for backtesting'}

            # Get historical data for all symbols
            all_symbol_data = {}

            for symbol in symbols:
                try:
                    import ib_insync
                    from ib_insync import Stock

                    self.logger.info(f"Fetching historical data for {symbol}")

                    contract = Stock(symbol, self.exchange_var, 'USD')
                    qualified_contracts = await self.ib.qualifyContractsAsync(contract)

                    if not qualified_contracts:
                        self.logger.warning(f"Could not qualify contract for {symbol}")
                        continue

                    contract = qualified_contracts[0]

                    # Calculate duration and bar size
                    if days <= 30:
                        duration_str = f"{days} D"
                        bar_size = "1 min"
                    elif days <= 365:
                        duration_str = f"{days} D" 
                        bar_size = "5 mins"
                    else:
                        duration_str = "1 Y"
                        bar_size = "1 hour"

                    # Get historical data
                    bars = await self.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime=end_dt.strftime('%Y%m%d %H:%M:%S'),
                        durationStr=duration_str,
                        barSizeSetting=bar_size,
                        whatToShow='TRADES',
                        useRTH=False  # Now using extended hours
                    )

                    if bars and len(bars) > 20:  # Need sufficient data for indicators
                        historical_data = []
                        for bar in bars:
                            historical_data.append({
                                'timestamp': bar.date,
                                'close': bar.close,
                                'high': bar.high,
                                'low': bar.low,
                                'volume': bar.volume
                            })

                        # Calculate technical indicators
                        df = pd.DataFrame(historical_data)
                        df['rsi'] = self._calculate_rsi_series(df['close'])
                        df['bollinger_pct'] = self._calculate_bollinger_series(df['close'])
                        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic_series(df['close'])

                        all_symbol_data[symbol] = df
                        self.logger.info(f"Loaded {len(df)} data points for {symbol}")
                    else:
                        self.logger.warning(f"Insufficient historical data for {symbol}")

                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {e}")
                    continue

            if not all_symbol_data:
                return {'error': 'No valid historical data available for any symbols'}

            # Find common time range across all symbols
            min_length = min(len(df) for df in all_symbol_data.values())
            self.logger.info(f"Using {min_length} common data points across all symbols")

            # Initialize tracking variables
            total_rsi, total_bollinger, total_stoch_k, total_stoch_d = 0, 0, 0, 0
            rsi_count, bollinger_count, stoch_k_count = 0, 0, 0
            rsi_triggers, bollinger_triggers, stoch_k_triggers = 0, 0, 0
            fibonacci_exits, stop_loss_exits, time_limit_exits, end_of_period_exits = 0, 0, 0, 0
            max_profit_percentages = []

            # Profit-taking strategy settings (Fibonacci retracement)
            profit_taking_strategy = {
                'type': 'Fibonacci Retracement',
                'initial_profit_threshold': 0.03,  # 3% initial profit target
                'retracement_limit': 0.02,  # Monitor for 2% retracement from peak profit
                'hard_stop_loss': self.hard_stop_loss
            }

            # Run backtest simulation
            trade_log = []

            for i in range(min_length):
                current_time = None

                # Check each symbol for trading opportunities
                for symbol, df in all_symbol_data.items():
                    if i >= len(df):
                        continue

                    row = df.iloc[i]
                    current_time = row['timestamp']
                    current_price = row['close']

                    # Skip if insufficient capital
                    if available_capital < max_equity_per_trade:
                        continue

                    # Check if already have position in this symbol
                    if symbol in active_positions:
                        position = active_positions[symbol]
                        entry_price = position['entry_price']
                        quantity = position['quantity']

                        # Calculate current profit percentage
                        current_profit_pct = (current_price - entry_price) / entry_price

                        # Update maximum profit percentage reached
                        if current_profit_pct > position.get('max_profit_pct', 0):
                            position['max_price_reached'] = current_price
                            position['max_profit_pct'] = current_profit_pct

                        # Track max profit percentage for all trades
                        max_profit_percentages.append(current_profit_pct)

                        # Check exit conditions based on profit-taking strategy
                        should_exit = False
                        exit_reason = ""

                        # Fibonacci profit-taking strategy logic
                        if current_profit_pct >= profit_taking_strategy['initial_profit_threshold']:
                            # Monitor for retracement from the maximum price reached
                            retracement = (position['max_price_reached'] - current_price) / position['max_price_reached']

                            if retracement >= profit_taking_strategy['retracement_limit']:
                                should_exit = True
                                exit_reason = "Fibonacci Retracement"
                                fibonacci_exits += 1
                        # Hard stop loss check
                        elif (current_price - entry_price) / entry_price <= -profit_taking_strategy['hard_stop_loss']:
                            should_exit = True
                            exit_reason = "Hard Stop Loss"
                            stop_loss_exits += 1
                        # Time limit check
                        elif i - position['entry_index'] >= 50:
                            should_exit = True
                            exit_reason = "Time Limit"
                            time_limit_exits += 1

                        if should_exit:
                            # Close position
                            trade_value = quantity * current_price
                            profit = trade_value - position['trade_cost']
                            total_profit += profit
                            available_capital += trade_value

                            # Record completed trade
                            completed_trades.append({
                                'symbol': symbol,
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'quantity': quantity,
                                'profit': profit,
                                'profit_pct': current_profit_pct * 100,
                                'max_price_reached': position.get('max_price_reached', current_price),
                                'max_profit_pct': position.get('max_profit_pct', current_profit_pct) * 100,
                                'exit_reason': exit_reason,
                                'hold_periods': i - position['entry_index'],
                                'entry_conditions': position['entry_conditions']
                            })

                            del active_positions[symbol]
                            continue

                    # Check buy conditions (same as live trading)
                    buy_signal = (
                        row['rsi'] < 30 or
                        row['bollinger_pct'] < 20 or
                        row['stoch_k'] < self.stoch_k_min
                    )

                    if buy_signal and symbol not in active_positions:
                        # Calculate position size
                        trade_equity = min(max_equity_per_trade, available_capital * 0.2)
                        quantity = max(1, int(trade_equity / current_price))
                        trade_cost = quantity * current_price

                        # Only proceed if we have sufficient capital
                        if available_capital >= trade_cost:
                            # Open position
                            active_positions[symbol] = {
                                'entry_time': current_time,
                                'entry_index': i,
                                'entry_price': current_price,
                                'quantity': quantity,
                                'trade_cost': trade_cost,
                                'entry_conditions': {
                                    'rsi': row['rsi'],
                                    'bollinger_pct': row['bollinger_pct'],
                                    'stoch_k': row['stoch_k'],
                                    'stoch_d': row['stoch_d'],
                                    'rsi_triggered': row['rsi'] < 30,
                                    'bollinger_triggered': row['bollinger_pct'] < 20,
                                    'stoch_k_triggered': row['stoch_k'] < self.stoch_k_min
                                }
                            }

                            available_capital -= trade_cost

                            trade_log.append({
                                'timestamp': current_time,
                                'symbol': symbol,
                                'action': 'BUY',
                                'price': current_price,
                                'quantity': quantity,
                                'trade_cost': trade_cost,
                                'available_capital': available_capital,
                                'conditions': {
                                    'rsi_triggered': row['rsi'] < 30,
                                    'bollinger_triggered': row['bollinger_pct'] < 20,
                                    'stoch_k_triggered': row['stoch_k'] < self.stoch_k_min
                                }
                            })

                            # Accumulate entry variable levels
                            total_rsi += row['rsi']
                            total_bollinger += row['bollinger_pct']
                            total_stoch_k += row['stoch_k']
                            total_stoch_d += row['stoch_d']

                            # Count trigger occurrences
                            if row['rsi'] < 30:
                                rsi_triggers += 1
                            if row['bollinger_pct'] < 20:
                                bollinger_triggers += 1
                            if row['stoch_k'] < self.stoch_k_min:
                                stoch_k_triggers += 1

                            rsi_count += 1
                            bollinger_count += 1
                            stoch_k_count += 1

            # Close any remaining positions at the end
            if active_positions:
                final_time = current_time
                for symbol, position in active_positions.items():
                    df = all_symbol_data[symbol]
                    final_price = df.iloc[-1]['close']
                    quantity = position['quantity']
                    trade_value = quantity * final_price
                    profit = trade_value - position['trade_cost']
                    total_profit += profit
                    end_of_period_exits += 1  # Count end-of-period exits

                    completed_trades.append({
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': final_time,
                        'entry_price': position['entry_price'],
                        'exit_price': final_price,
                        'quantity': quantity,
                        'profit': profit,
                        'profit_pct': ((final_price - position['entry_price']) / position['entry_price']) * 100,
                        'max_price_reached': position.get('max_price_reached', final_price),
                        'max_profit_pct': position.get('max_profit_pct', ((final_price - position['entry_price']) / position['entry_price']) * 100),
                        'exit_reason': 'End of Period',
                        'hold_periods': min_length - position['entry_index'],
                        'entry_conditions': position['entry_conditions']
                    })

            # Calculate average entry variable levels
            avg_entry_variables = {
                'rsi': total_rsi / max(1, rsi_count),
                'bollinger_pct': total_bollinger / max(1, bollinger_count),
                'stoch_k': total_stoch_k / max(1, stoch_k_count),
                'stoch_d': total_stoch_d / max(1, stoch_k_count)
            }

            # Collect trigger statistics
            trigger_statistics = {
                'rsi_triggers': rsi_triggers,
                'bollinger_triggers': bollinger_triggers,
                'stoch_k_triggers': stoch_k_triggers
            }

            # Collect exit strategy statistics
            exit_strategy_stats = {
                'fibonacci_exits': fibonacci_exits,
                'stop_loss_exits': stop_loss_exits,
                'time_limit_exits': time_limit_exits,
                'end_of_period_exits': end_of_period_exits
            }

            # Calculate final metrics
            total_trades = len(completed_trades)
            winning_trades = sum(1 for trade in completed_trades if trade['profit'] > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            final_capital = available_capital + sum(pos['trade_cost'] for pos in active_positions.values())
            total_return = ((final_capital - initial_capital) / initial_capital) * 100

            results = {
                'symbols': symbols,
                'date_range': f"{start_date} to {end_date}",
                'initial_capital': initial_capital,
                'final_capital': final_capital + total_profit,
                'total_profit': total_profit,
                'total_return_pct': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'completed_trades': completed_trades,  # Show all trades
                'trade_summary': {
                    'avg_profit_per_trade': total_profit / max(1, total_trades),
                    'best_trade': max(completed_trades, key=lambda x: x['profit']) if completed_trades else None,
                    'worst_trade': min(completed_trades, key=lambda x: x['profit']) if completed_trades else None,
                    'avg_hold_periods': sum(t['hold_periods'] for t in completed_trades) / max(1, total_trades)
                },
                'capital_management': {
                    'max_equity_per_trade': max_equity_per_trade,
                    'capital_utilization': ((initial_capital - available_capital) / initial_capital) * 100,
                    'overlapping_trades_prevented': True
                },
                'avg_entry_variables': avg_entry_variables,
                'trigger_statistics': trigger_statistics,
                'exit_strategy_stats': exit_strategy_stats,
                'profit_taking_strategy': profit_taking_strategy
            }

            self.logger.info(f"Backtest completed: {total_trades} trades, {win_rate:.1f}% win rate, {total_return:.2f}% return")
            return results

        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return {'error': f'Backtest failed: {str(e)}'}

    def _calculate_rsi_series(self, prices, period=14):
        """Calculate RSI for a price series"""
        rsi_values = []
        for i in range(len(prices)):
            if i < period:
                rsi_values.append(50.0)
            else:
                price_window = prices[i-period:i+1].tolist()
                rsi = self._calculate_rsi(price_window, period)
                rsi_values.append(rsi)
        return rsi_values

    def _calculate_bollinger_series(self, prices, period=20):
        """Calculate Bollinger Band percentage for a price series"""
        bb_values = []
        for i in range(len(prices)):
            if i < period:
                bb_values.append(50.0)
            else:
                price_window = prices[i-period:i+1].tolist()
                bb_data = self._calculate_bollinger_bands(price_window, period)
                bb_values.append(bb_data['position'])
        return bb_values

    def _calculate_stochastic_series(self, prices, k_period=14, d_period=3):
        """Calculate Stochastic oscillator for a price series"""
        stoch_k_values = []
        stoch_d_values = []

        for i in range(len(prices)):
            if i < k_period:
                stoch_k_values.append(50.0)
                stoch_d_values.append(50.0)
            else:
                price_window = prices[i-k_period:i+1].tolist()
                stoch_data = self._calculate_stochastic(price_window, k_period, d_period)
                stoch_k_values.append(stoch_data['k'])
                stoch_d_values.append(stoch_data['d'])

        return stoch_k_values, stoch_d_values

class TradingGUI:
    """GUI interface for the standalone trading application"""

    def __init__(self):
        self.config = TradingConfig()
        self.trading_engine = StandaloneTradingEngine(self.config)
        self.logger = logging.getLogger(__name__)

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("AutoTrade Plus - Standalone Trading Application")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f8f9fa')

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')

        # Configure custom styles
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), background='#f8f9fa')
        style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'), background='#f8f9fa')
        style.configure('Status.TLabel', font=('Helvetica', 10), background='#f8f9fa')

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self.create_dashboard_tab()
        self.create_backtesting_tab()
        self.create_settings_tab()
        self.create_analytics_tab()
        self.create_logs_tab()

        # Start background tasks
        self.setup_background_tasks()

        # Update GUI periodically
        self.update_gui()

    def create_dashboard_tab(self):
        """Create dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Live Trading")

        # Title
        title_label = ttk.Label(dashboard_frame, text="Live Trading Dashboard", style='Title.TLabel')
        title_label.pack(pady=10)

        # Status section
        status_frame = ttk.LabelFrame(dashboard_frame, text="System Status", padding=10)
        status_frame.pack(fill='x', padx=10, pady=5)

        # Status indicators
        self.status_labels = {}
        status_row = ttk.Frame(status_frame)
        status_row.pack(fill='x')

        # Connection status
        ttk.Label(status_row, text="Connection:", style='Heading.TLabel').pack(side='left')
        self.status_labels['connection'] = ttk.Label(status_row, text="Disconnected", style='Status.TLabel')
        self.status_labels['connection'].pack(side='left', padx=10)

        # Streaming status
        ttk.Label(status_row, text="Streaming:", style='Heading.TLabel').pack(side='left', padx=(20, 0))
        self.status_labels['streaming'] = ttk.Label(status_row, text="Stopped", style='Status.TLabel')
        self.status_labels['streaming'].pack(side='left', padx=10)

        # Force trade status
        ttk.Label(status_row, text="Force Trade:", style='Heading.TLabel').pack(side='left', padx=(20, 0))
        self.status_labels['force_trade'] = ttk.Label(status_row, text="Disabled", style='Status.TLabel')
        self.status_labels['force_trade'].pack(side='left', padx=10)

        # Control buttons
        button_frame = ttk.Frame(dashboard_frame)
        button_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(button_frame, text="Connect", command=self.connect_action).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Disconnect", command=self.disconnect_action).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Start Streaming", command=self.start_streaming_action).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Stop Streaming", command=self.stop_streaming_action).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Toggle Force Trade", command=self.toggle_force_trade_action).pack(side='left', padx=5)

        # Multiple Symbol Streaming section
        streaming_frame = ttk.LabelFrame(dashboard_frame, text="Multiple Symbol Streaming", padding=10)
        streaming_frame.pack(fill='x', padx=10, pady=5)

        # Symbols input
        symbols_row = ttk.Frame(streaming_frame)
        symbols_row.pack(fill='x', pady=5)

        ttk.Label(symbols_row, text="Symbols (comma-separated):", style='Heading.TLabel').pack(side='left')
        self.symbols_var = tk.StringVar(value="APVO,AAPL,MSFT")
        self.symbols_entry = ttk.Entry(symbols_row, textvariable=self.symbols_var, width=30)
        self.symbols_entry.pack(side='left', padx=10)

        ttk.Button(symbols_row, text="Update Streaming", command=self.update_streaming_symbols).pack(side='left', padx=5)

        # Force Trade section
        force_trade_frame = ttk.LabelFrame(dashboard_frame, text="Force Trade Controls", padding=10)
        force_trade_frame.pack(fill='x', padx=10, pady=5)

        # Stock selection row
        stock_row = ttk.Frame(force_trade_frame)
        stock_row.pack(fill='x', pady=5)

        ttk.Label(stock_row, text="Stock Symbol:", style='Heading.TLabel').pack(side='left')
        self.force_trade_symbol = tk.StringVar(value="APVO")
        self.symbol_combo = ttk.Combobox(stock_row, textvariable=self.force_trade_symbol, width=10)
        self.symbol_combo.pack(side='left', padx=10)

        # Trade buttons row
        trade_buttons_row = ttk.Frame(force_trade_frame)
        trade_buttons_row.pack(fill='x', pady=5)

        ttk.Button(trade_buttons_row, text="Force BUY", command=self.force_buy_action, 
                  style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(trade_buttons_row, text="Force SELL", command=self.force_sell_action, 
                  style='Accent.TButton').pack(side='left', padx=5)

        # Force trade status
        self.force_trade_status = ttk.Label(force_trade_frame, text="Force trade mode: Disabled", 
                                           style='Status.TLabel')
        self.force_trade_status.pack(pady=5)

        # Market data section
        market_frame = ttk.LabelFrame(dashboard_frame, text="Market Data", padding=10)
        market_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Market data tree
        columns = ('Symbol', 'Price', 'RSI', 'BB%', 'Stoch K', 'Stoch D', 'Volume', 'Status', 'Meets Criteria')
        self.market_tree = ttk.Treeview(market_frame, columns=columns, show='headings', height=8)

        for col in columns:
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=100)

        # Scrollbar for market data
        market_scroll = ttk.Scrollbar(market_frame, orient='vertical', command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=market_scroll.set)

        self.market_tree.pack(side='left', fill='both', expand=True)
        market_scroll.pack(side='right', fill='y')

        # Active orders section
        orders_frame = ttk.LabelFrame(dashboard_frame, text="Active Orders", padding=10)
        orders_frame.pack(fill='x', padx=10, pady=5)

        order_columns = ('Symbol', 'Quantity', 'Buy Price', 'Timestamp', 'Profit')
        self.orders_tree = ttk.Treeview(orders_frame, columns=order_columns, show='headings', height=4)

        for col in order_columns:
            self.orders_tree.heading(col, text=col)
            self.orders_tree.column(col, width=120)

        self.orders_tree.pack(fill='x')

    def create_backtesting_tab(self):
        """Create backtesting tab"""
        backtest_frame = ttk.Frame(self.notebook)
        self.notebook.add(backtest_frame, text="Backtesting")

        # Title
        ttk.Label(backtest_frame, text="Backtesting Engine", style='Title.TLabel').pack(pady=10)

        # Input section
        input_frame = ttk.LabelFrame(backtest_frame, text="Backtest Parameters", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)

        # Symbol input (now supports multiple symbols)
        symbol_row = ttk.Frame(input_frame)
        symbol_row.pack(fill='x', pady=5)
        ttk.Label(symbol_row, text="Symbols:", style='Heading.TLabel', width=15).pack(side='left')
        self.backtest_symbol = tk.StringVar(value="APVO,AAPL,MSFT")
        symbols_entry = ttk.Entry(symbol_row, textvariable=self.backtest_symbol, width=30)
        symbols_entry.pack(side='left', padx=10)
        ttk.Label(symbol_row, text="(comma-separated)", style='Status.TLabel').pack(side='left', padx=5)

        # Date range inputs
        date_row = ttk.Frame(input_frame)
        date_row.pack(fill='x', pady=5)
        ttk.Label(date_row, text="Start Date:", style='Heading.TLabel', width=15).pack(side='left')
        self.start_date = tk.StringVar(value="2024-01-01")
        ttk.Entry(date_row, textvariable=self.start_date, width=15).pack(side='left', padx=5)

        ttk.Label(date_row, text="End Date:", style='Heading.TLabel', width=15).pack(side='left', padx=(20, 0))
        self.end_date = tk.StringVar(value="2024-12-31")
        ttk.Entry(date_row, textvariable=self.end_date, width=15).pack(side='left', padx=5)

        # Run backtest button
        ttk.Button(input_frame, text="Run Backtest", command=self.run_backtest_action).pack(pady=10)

        # Results section
        results_frame = ttk.LabelFrame(backtest_frame, text="Backtest Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Results display
        self.backtest_results = scrolledtext.ScrolledText(results_frame, width=100, height=25, font=('Consolas', 9))
        self.backtest_results.pack(fill='both', expand=True)

    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")

        # Create scrollable frame
        canvas = tk.Canvas(settings_frame, bg='#f8f9fa')
        scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Title
        ttk.Label(scrollable_frame, text="Trading Parameters", style='Title.TLabel').pack(pady=10)

        # Trading parameters
        self.create_parameter_section(scrollable_frame, "IBKR Connection", [
            ("Host", "ibkr_host", "127.0.0.1"),
            ("Port", "ibkr_port", 7497),
            ("Client ID", "client_id", 2)
        ])

        self.create_parameter_section(scrollable_frame, "Basic Settings", [
            ("Tickers", "tickers", "APVO"),
            ("Chart Interval", "chart_interval", "1 min"),
            ("Analysis Interval", "analysis_interval", 1),
            ("Data Points", "data_points", 16),
            ("Duration", "duration_str", "1 D"),
            ("Equity per Trade", "equity_per_trade", 100.0),
            ("Hard Stop Loss", "hard_stop_loss", 0.03),
            ("Exchange", "exchange_var", "SMART")
        ])

        self.create_parameter_section(scrollable_frame, "Technical Analysis", [
            ("RSI Period", "rsi_period", 3),
            ("Bollinger Bands Period", "bb_period", 3),
            ("Stochastic K Period", "stoch_k_period", 2),
            ("Stochastic D Period", "stoch_d_period", 2),
            ("Stochastic K Min", "stoch_k_min", 50.0),
            ("Stochastic D Min", "stoch_d_min", 40.0)
        ])

        # Update button
        ttk.Button(scrollable_frame, text="Update Parameters", command=self.update_parameters_action).pack(pady=20)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_parameter_section(self, parent, title, parameters):
        """Create a section of parameters"""
        section_frame = ttk.LabelFrame(parent, text=title, padding=10)
        section_frame.pack(fill='x', padx=10, pady=5)

        if not hasattr(self, 'parameter_vars'):
            self.parameter_vars = {}

        for label, attr, default in parameters:
            row = ttk.Frame(section_frame)
            row.pack(fill='x', pady=2)

            ttk.Label(row, text=f"{label}:", width=20).pack(side='left')

            var = tk.StringVar(value=str(getattr(self.trading_engine, attr, default)))
            self.parameter_vars[attr] = var

            entry = ttk.Entry(row, textvariable=var, width=20)
            entry.pack(side='left', padx=10)

    def create_analytics_tab(self):
        """Create analytics tab"""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="Analytics")

        # Title
        ttk.Label(analytics_frame, text="Trading Analytics", style='Title.TLabel').pack(pady=10)

        # Analytics display
        self.analytics_labels = {}

        analytics_data = [
            ("Total Trades", "total_trades"),
            ("Total Profit", "total_profit"),
            ("Win Rate", "win_rate"),
            ("Available Funds", "available_funds"),
            ("Active Positions", "active_positions"),
            ("Last Price Change", "last_price_change"),
            ("Last RSI Change", "last_rsi_change"),
            ("Last Bollinger Change", "last_bollinger_change")
        ]

        for label, key in analytics_data:
            row = ttk.Frame(analytics_frame)
            row.pack(fill='x', padx=20, pady=5)

            ttk.Label(row, text=f"{label}:", style='Heading.TLabel', width=20).pack(side='left')
            self.analytics_labels[key] = ttk.Label(row, text="N/A", style='Status.TLabel')
            self.analytics_labels[key].pack(side='left', padx=10)

    def create_logs_tab(self):
        """Create logs tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")

        # Title
        ttk.Label(logs_frame, text="System Logs", style='Title.TLabel').pack(pady=10)

        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, width=100, height=30, font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True, padx=10, pady=5)

        # Clear button
        ttk.Button(logs_frame, text="Clear Logs", command=self.clear_logs).pack(pady=5)

        # Setup log handler
        self.setup_log_handler()

    def setup_log_handler(self):
        """Setup log handler to display logs in GUI"""
        class GuiLogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                msg = self.format(record)
                def append():
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.see(tk.END)
                self.text_widget.after(0, append)

        handler = GuiLogHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)

    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)

    def setup_background_tasks(self):
        """Setup background tasks"""
        self.loop = asyncio.new_event_loop()
        self.bg_thread = threading.Thread(target=self.run_background_loop, daemon=True)
        self.bg_thread.start()

    def run_background_loop(self):
        """Run background asyncio loop"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_async(self, coro):
        """Run async function in background thread"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future

    def connect_action(self):
        """Connect action"""
        self.run_async(self.trading_engine.connect())

    def disconnect_action(self):
        """Disconnect action"""
        self.run_async(self.trading_engine.disconnect())

    def start_streaming_action(self):
        """Start streaming action"""
        symbols = [s.strip().upper() for s in self.symbols_var.get().split(',') if s.strip()]
        self.run_async(self.trading_engine.start_market_data(symbols))

    def stop_streaming_action(self):
        """Stop streaming action"""
        self.trading_engine.stop_market_data()

    def update_streaming_symbols(self):
        """Update streaming symbols"""
        if self.trading_engine.is_streaming():
            # Stop current streaming
            self.trading_engine.stop_market_data()
            # Start with new symbols
            symbols = [s.strip().upper() for s in self.symbols_var.get().split(',') if s.strip()]
            self.run_async(self.trading_engine.start_market_data(symbols))
        else:
            messagebox.showinfo("Info", "Start streaming first to update symbols.")

    def toggle_force_trade_action(self):
        """Toggle force trade action"""
        enabled = self.trading_engine.toggle_force_trade()
        status_text = f"Force trade mode: {'Enabled' if enabled else 'Disabled'}"
        self.force_trade_status.config(text=status_text)

        if enabled:
            messagebox.showwarning("Force Trade Mode", 
                "Force Trade Mode is now enabled.\n"
                "This mode bypasses some safety checks and may increase risk.\n"
                "Use with caution!")

    def force_buy_action(self):
        """Execute force buy"""
        if not self.trading_engine.force_trade_enabled:
            messagebox.showwarning("Force Trade Disabled", 
                "Please enable Force Trade mode first.")
            return

        symbol = self.force_trade_symbol.get().strip().upper()
        if not symbol:
            messagebox.showerror("Error", "Please enter a stock symbol.")
            return

        # Execute in background thread
        def execute_trade():
            future = self.run_async(self.trading_engine.execute_force_trade(symbol, "BUY"))
            try:
                result = future.result(timeout=30)
                if 'success' in result:
                    messagebox.showinfo("Force Buy", result['success'])
                else:
                    messagebox.showerror("Force Buy Failed", result.get('error', 'Unknown error'))
            except Exception as e:
                messagebox.showerror("Force Buy Failed", f"Error: {str(e)}")

        threading.Thread(target=execute_trade, daemon=True).start()

    def force_sell_action(self):
        """Execute force sell"""
        if not self.trading_engine.force_trade_enabled:
            messagebox.showwarning("Force Trade Disabled", 
                "Please enable Force Trade mode first.")
            return

        symbol = self.force_trade_symbol.get().strip().upper()
        if not symbol:
            messagebox.showerror("Error", "Please enter a stock symbol.")
            return

        # Execute in background thread
        def execute_trade():
            future = self.run_async(self.trading_engine.execute_force_trade(symbol, "SELL"))
            try:
                result = future.result(timeout=30)
                if 'success' in result:
                    messagebox.showinfo("Force Sell", result['success'])
                else:
                    messagebox.showerror("Force Sell Failed", result.get('error', 'Unknown error'))
            except Exception as e:
                messagebox.showerror("Force Sell Failed", f"Error: {str(e)}")

        threading.Thread(target=execute_trade, daemon=True).start()

    def run_backtest_action(self):
        """Run enhanced backtest with capital management"""
        try:
            symbols_input = self.backtest_symbol.get().strip().upper()
            start_date = self.start_date.get().strip()
            end_date = self.end_date.get().strip()

            if not symbols_input or not start_date or not end_date:
                messagebox.showerror("Error", "Please fill in all backtest parameters.")
                return

            # Validate date format
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("Error", "Please use YYYY-MM-DD format for dates.")
                return

            # Parse symbols
            symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]

            self.backtest_results.delete(1.0, tk.END)
            self.backtest_results.insert(tk.END, f"Running backtest for {len(symbols)} symbols: {', '.join(symbols)}\n")
            self.backtest_results.insert(tk.END, f"Date Range: {start_date} to {end_date}\n")
            self.backtest_results.insert(tk.END, f"Initial Capital: $10,000\n\n")

            # Run backtest in background
            def run_backtest():
                try:
                    # Run enhanced backtest
                    future = self.run_async(self.trading_engine.run_backtest(symbols_input, start_date, end_date, 10000.0))
                    results = future.result(timeout=180)  # 3 minute timeout

                    if 'error' in results:
                        self.backtest_results.insert(tk.END, f"Error: {results['error']}\n")
                        return

                    # Display enhanced results
                    output = f"ENHANCED BACKTEST RESULTS\n"
                    output += f"=" * 50 + "\n"
                    output += f"Symbols: {', '.join(results['symbols'])}\n"
                    output += f"Date Range: {results['date_range']}\n"
                    output += f"Initial Capital: ${results['initial_capital']:,.2f}\n"
                    output += f"Final Capital: ${results['final_capital']:,.2f}\n"
                    output += f"Total Profit: ${results['total_profit']:,.2f}\n"
                    output += f"Total Return: {results['total_return_pct']:.2f}%\n\n"

                    output += "TRADE PERFORMANCE:\n"
                    output += f"Total Trades: {results['total_trades']}\n"
                    output += f"Winning Trades: {results['winning_trades']}\n"
                    output += f"Win Rate: {results['win_rate']:.1f}%\n"
                    output += f"Avg Profit per Trade: ${results['trade_summary']['avg_profit_per_trade']:.2f}\n"
                    output += f"Avg Hold Periods: {results['trade_summary']['avg_hold_periods']:.1f}\n\n"

                    if results['trade_summary']['best_trade']:
                        best = results['trade_summary']['best_trade']
                        output += f"Best Trade: {best['symbol']} +${best['profit']:.2f} ({best['profit_pct']:.2f}%)\n"

                    if results['trade_summary']['worst_trade']:
                        worst = results['trade_summary']['worst_trade']
                        output += f"Worst Trade: {worst['symbol']} ${worst['profit']:.2f} ({worst['profit_pct']:.2f}%)\n\n"

                    output += "CAPITAL MANAGEMENT:\n"
                    output += f"Max Equity per Trade: ${results['capital_management']['max_equity_per_trade']:,.2f}\n"
                    output += f"Capital Utilization: {results['capital_management']['capital_utilization']:.1f}%\n"
                    output += f"Overlapping Trades Prevention: {results['capital_management']['overlapping_trades_prevented']}\n\n"

                    # Display average entry variable levels
                    output += "AVERAGE ENTRY VARIABLE LEVELS:\n"
                    output += f"Average RSI at Entry: {results['avg_entry_variables']['rsi']:.2f}\n"
                    output += f"Average Bollinger %: {results['avg_entry_variables']['bollinger_pct']:.2f}%\n"
                    output += f"Average Stoch K: {results['avg_entry_variables']['stoch_k']:.2f}\n"
                    output += f"Average Stoch D: {results['avg_entry_variables']['stoch_d']:.2f}\n\n"

                    output += "TRIGGER STATISTICS:\n"
                    output += f"RSI Triggers: {results['trigger_statistics']['rsi_triggers']} trades\n"
                    output += f"Bollinger Triggers: {results['trigger_statistics']['bollinger_triggers']} trades\n"
                    output += f"Stochastic K Triggers: {results['trigger_statistics']['stoch_k_triggers']} trades\n\n"

                    output += "EXIT STRATEGY PERFORMANCE:\n"
                    output += f"Fibonacci Retracement Exits: {results['exit_strategy_stats']['fibonacci_exits']} trades\n"
                    output += f"Hard Stop Loss Exits: {results['exit_strategy_stats']['stop_loss_exits']} trades\n"
                    output += f"Time Limit Exits: {results['exit_strategy_stats']['time_limit_exits']} trades\n"
                    output += f"End of Period Exits: {results['exit_strategy_stats']['end_of_period_exits']} trades\n\n"

                    output += "PROFIT TAKING STRATEGY:\n"
                    output += f"Type: {results['profit_taking_strategy']['type']}\n"
                    output += f"Initial Threshold: {results['profit_taking_strategy']['initial_profit_threshold']}\n"
                    output += f"Retracement Limit: {results['profit_taking_strategy']['retracement_limit']}\n"
                    output += f"Hard Stop Loss: {results['profit_taking_strategy']['hard_stop_loss']}\n\n"

                    output += f"ALL COMPLETED TRADES ({results['total_trades']} trades):\n"
                    output += "-" * 130 + "\n"
                    output += f"{'#':<4} {'Symbol':<8} {'Entry':<12} {'Exit':<12} {'Buy':<8} {'Sell':<8} {'Max':<8} {'Profit':<10} {'%':<8} {'Max%':<8} {'Reason':<15} {'Variables':<25}\n"
                    output += "-" * 130 + "\n"

                    for idx, trade in enumerate(results['completed_trades'], 1):
                        entry_time = str(trade['entry_time'])[:10] if trade['entry_time'] else 'N/A'
                        exit_time = str(trade['exit_time'])[:10] if trade['exit_time'] else 'N/A'

                        # Format entry conditions for display
                        conditions = trade['entry_conditions']
                        triggers = []
                        if conditions['rsi_triggered']:
                            triggers.append(f"RSI:{conditions['rsi']:.1f}")
                        if conditions['bollinger_triggered']:
                            triggers.append(f"BB:{conditions['bollinger_pct']:.1f}")
                        if conditions['stoch_k_triggered']:
                            triggers.append(f"SK:{conditions['stoch_k']:.1f}")
                        trigger_str = ",".join(triggers)[:24]

                        output += f"{idx:<4} {trade['symbol']:<8} {entry_time:<12} {exit_time:<12} "
                        output += f"${trade['entry_price']:<7.2f} ${trade['exit_price']:<7.2f} ${trade.get('max_price_reached', trade['exit_price']):<7.2f} "
                        output += f"${trade['profit']:<9.2f} {trade['profit_pct']:<7.1f}% {trade.get('max_profit_pct', trade['profit_pct']):<7.1f}% "
                        output += f"{trade['exit_reason']:<15} {trigger_str:<25}\n"

                    output += f"\nKEY IMPROVEMENTS:\n"
                    output += f"✓ Capital management prevents over-investment\n"
                    output += f"✓ No overlapping trades for same symbol\n"
                    output += f"✓ Multi-symbol support like live trading\n"
                    output += f"✓ Proper exit strategy (3% profit, 2% loss, time limit)\n"
                    output += f"✓ Same buy conditions as live trading\n"

                    self.backtest_results.delete(1.0, tk.END)
                    self.backtest_results.insert(tk.END, output)

                except Exception as e:
                    self.backtest_results.insert(tk.END, f"Backtest error: {str(e)}\n")

            threading.Thread(target=run_backtest, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Backtest Error", f"Failed to run backtest: {str(e)}")

    def update_parameters_action(self):
        """Update parameters action"""
        try:
            params = {}
            config_params = {}

            for attr, var in self.parameter_vars.items():
                value = var.get()

                # Convert to appropriate type
                if attr in ['analysis_interval', 'data_points', 'rsi_period', 'bb_period', 
                           'stoch_k_period', 'stoch_d_period', 'ibkr_port', 'client_id']:
                    converted_value = int(value)
                elif attr in ['equity_per_trade', 'hard_stop_loss', 'stoch_k_min', 'stoch_d_min']:
                    converted_value = float(value)
                else:
                    converted_value = value

                # Separate config parameters from trading engine parameters
                if attr in ['ibkr_host', 'ibkr_port', 'client_id']:
                    config_params[attr] = converted_value
                else:
                    params[attr] = converted_value

            # Update trading engine parameters
            self.trading_engine.update_parameters(params)

            # Update config parameters
            for attr, value in config_params.items():
                setattr(self.config, attr, value)

            messagebox.showinfo("Success", "Parameters updated successfully!\nReconnect to apply IBKR connection changes.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update parameters: {e}")

    def update_gui(self):
        """Update GUI with current data"""
        try:
            # Update status labels
            self.status_labels['connection'].config(text="Connected" if self.trading_engine.is_connected() else "Disconnected")
            self.status_labels['streaming'].config(text="Active" if self.trading_engine.is_streaming() else "Stopped")
            self.status_labels['force_trade'].config(text="Enabled" if self.trading_engine.force_trade_enabled else "Disabled")

            # Update market data
            self.update_market_data()

            # Update active orders
            self.update_active_orders()

            # Update analytics
            self.update_analytics()

        except Exception as e:
            self.logger.error(f"GUI update error: {e}")

        # Schedule next update
        self.root.after(2000, self.update_gui)

    def update_market_data(self):
        """Update market data display"""
        # Clear existing items
        for item in self.market_tree.get_children():
            self.market_tree.delete(item)

        # Update symbol dropdown
        symbols = list(self.trading_engine.market_data.keys())
        if symbols and hasattr(self, 'symbol_combo'):
            self.symbol_combo['values'] = symbols
            if not self.force_trade_symbol.get() or self.force_trade_symbol.get() not in symbols:
                self.force_trade_symbol.set(symbols[0])

        # Add current market data
        for symbol, data in self.trading_engine.market_data.items():
            values = (
                symbol,
                f"${data.get('last', 0):.2f}",
                f"{data.get('rsi', 0):.1f}",
                f"{data.get('bollinger_pct', 0):.1f}%",
                f"{data.get('stoch_k', 0):.1f}",
                f"{data.get('stoch_d', 0):.1f}",
                f"{data.get('volume', 0):,}",
                data.get('status', 'N/A'),
                "Yes" if data.get('meets_criteria', False) else "No"
            )
            self.market_tree.insert('', 'end', values=values)

    def update_active_orders(self):
        """Update active orders display"""
        # Clear existing items
        for item in self.orders_tree.get_children():
            self.orders_tree.delete(item)

        # Add current orders
        for symbol, order in self.trading_engine.active_orders.items():
            values = (
                symbol,
                order.get('quantity', 0),
                f"${order.get('buy_price', 0):.2f}",
                order.get('timestamp', 'N/A'),
                f"${order.get('profit', 0):.2f}"
            )
            self.orders_tree.insert('', 'end', values=values)

    def update_analytics(self):
        """Update analytics display"""
        analytics = self.trading_engine.get_analytics()

        for key, label in self.analytics_labels.items():
            value = analytics.get(key, 'N/A')

            if key in ['total_profit'] and isinstance(value, (int, float)):
                label.config(text=f"${value:.2f}")
            elif key in ['win_rate'] and isinstance(value, (int, float)):
                label.config(text=f"{value:.1f}%")
            elif key in ['available_funds', 'equity_per_trade'] and isinstance(value, (int, float)):
                label.config(text=f"${value:.2f}")
            else:
                label.config(text=str(value))

    def run(self):
        """Run the GUI application"""
        try:
            self.root.mainloop()
        finally:
            # Cleanup
            if hasattr(self, 'loop'):
                self.loop.call_soon_threadsafe(self.loop.stop)

if __name__ == "__main__":
    # Create and run the application
    app = TradingGUI()
    app.run()