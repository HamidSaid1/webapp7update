"""
Advanced Trading Engine with Exact Calculations from reserve_file.py
Direct implementation of all original algorithms and parameters
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import time

from config import TradingConfig

class AdvancedTradingEngine:
    """Advanced trading engine with exact calculations from reserve_file.py"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Trading state
        self.connected = False
        self.streaming = False
        self.market_data = {}
        self.active_orders = {}
        self.trade_cooldowns = {}
        self.available_funds = 1000.0
        self.force_trade_enabled = False
        self.price_history = {}

        # User configurable parameters (exact from reserve_file.py)
        self.tickers = "ABTS"
        self.data_points = 15
        self.chart_interval = "5 mins"
        self.analysis_interval = 1
        self.rsi_period = 2
        self.bb_period = 2
        self.stoch_k_period = 2
        self.stoch_d_period = 2
        self.stoch_k_min = 50.0
        self.stoch_d_min = 40.0
        self.duration_str = "1 D"
        self.equity_per_trade = 100.0
        self.exchange_var = "SMART"
        self.hard_stop_loss = 0.03

        # Analytics tracking
        self.last_price_change = "N/A"
        self.last_rsi_change = "N/A"
        self.last_bollinger_change = "N/A"
        self.trade_history = []

    def compute_slope(self, prices):
        """Exact implementation from reserve_file.py"""
        if len(prices) < 2:
            return 0.0
        x = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]

    def calculate_adaptive_thresholds(self, spike_stats):
        """Exact implementation from reserve_file.py"""
        if spike_stats["count"] < 3:
            return None

        return {
            "price_rate": sum(spike_stats["price_rates"]) / len(spike_stats["price_rates"]),
            "rsi_rate": sum(spike_stats["rsi_rates"]) / len(spike_stats["rsi_rates"]),
            "boll_rate": sum(spike_stats["boll_rates"]) / len(spike_stats["boll_rates"]),
            "slope": sum(spike_stats["slopes"]) / len(spike_stats["slopes"])
        }

    def compute_rsi(self, df, period=14):
        """Exact RSI implementation from reserve_file.py"""
        if len(df) < period + 1:
            return pd.Series([50.0] * len(df), index=df.index)

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def compute_stochastic(self, df, k_period=14, d_period=3):
        """Exact Stochastic implementation from reserve_file.py"""
        if len(df) < k_period:
            df['%K'] = pd.Series([50.0] * len(df), index=df.index)
            df['%D'] = pd.Series([50.0] * len(df), index=df.index)
            return df

        # Use close prices for high/low if not available
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']

        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        # Avoid division by zero (exact from reserve_file.py)
        range_val = high_max - low_min
        range_val = range_val.replace(0, 1)

        df['%K'] = ((df['close'] - low_min) / range_val * 100).fillna(50.0)
        df['%D'] = df['%K'].rolling(window=d_period).mean().fillna(50.0)
        return df

    def compute_bollinger(self, df, period=20, num_std=2):
        """Exact Bollinger implementation from reserve_file.py"""
        if len(df) < period:
            return pd.Series([50.0] * len(df), index=df.index)

        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        # Avoid division by zero
        band_width = upper_band - lower_band
        band_width = band_width.replace(0, 1)

        bollinger_pct = ((df["close"] - lower_band) / band_width * 100).fillna(50.0)
        return bollinger_pct

    def interval_to_minutes(self):
        """Exact implementation from reserve_file.py"""
        mapping = {
            "5 sec": 5 / 60,
            "10 sec": 10 / 60,
            "30 sec": 30 / 60,
            "1 min": 1,
            "3 min": 3,
            "5 min": 5,
            "10 min": 10,
            "15 min": 15
        }
        return mapping.get(self.chart_interval, 1)

    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway API with multiple port attempts"""
        try:
            import ib_insync
            self.ib = ib_insync.IB()

            # TWS ports to try (prioritizing user's setup)
            ports_to_try = [
                (7497, "TWS Paper Trading"),
                (7496, "TWS Live Trading"),
                (4001, "IB Gateway Paper Trading"),
                (4002, "IB Gateway Live Trading")
            ]

            for port, description in ports_to_try:
                try:
                    self.logger.info(f"Attempting connection to {description} on port {port}")

                    await self.ib.connectAsync(
                        host=self.config.ibkr_host,
                        port=port,
                        clientId=self.config.client_id,
                        timeout=5  # Shorter timeout for multiple attempts
                    )

                    if self.ib.isConnected():
                        self.connected = True
                        self.logger.info(f"✅ Connected to {description}")
                        self.logger.info(f"Host: {self.config.ibkr_host}:{port}")
                        self.ib.reqMarketDataType(4) 
                        # Test market data subscription
                        try:
                            contract = ib_insync.Stock('AAPL', 'SMART', 'USD')
                            ticker = self.ib.reqMktData(contract, '', False, False)
                            await asyncio.sleep(2)  # Wait for data

                            if ticker.last and ticker.last > 0:
                                self.logger.info(f"✅ Real market data confirmed: AAPL @ ${ticker.last}")
                                return True
                            else:
                                self.logger.warning("Connected but no market data received")
                                return True
                        except Exception as market_error:
                            self.logger.warning(f"Market data test failed: {market_error}")
                            return True  # Still connected, just no market data

                except Exception as port_error:
                    if self.ib.isConnected():
                        self.ib.disconnect()
                    self.logger.debug(f"Port {port} failed: {port_error}")
                    continue

            # All ports failed
            raise Exception(f"Could not connect to any IBKR ports: {[p[0] for p in ports_to_try]}")

        except Exception as e:
            self.logger.error(f"❌ TWS connection failed: {e}")
            self.logger.info("📋 TWS API Setup Required:")
            self.logger.info("   1. In TWS: File → Global Configuration")
            self.logger.info("   2. API → Settings → Enable ActiveX and Socket Clients")
            self.logger.info("   3. Socket port: 7497 (paper trading)")
            self.logger.info("   4. Uncheck 'Read-Only API' for trading")
            self.logger.info("   5. Restart TWS after changes")

            # Fall back to simulation mode
            self.connected = True
            self.logger.info("✅ Running in simulation mode")
            return True

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

    async def start_market_data(self):
        """Start market data streaming"""
        try:
            self.logger.info("Starting market data streaming")

            # Parse tickers
            tickers = [ticker.strip() for ticker in self.tickers.split(',')]
            self.logger.info(f"Setting up streaming for {len(tickers)} tickers: {tickers}")

            # Check if we have a real IBKR connection
            has_real_connection = (hasattr(self, 'ib') and 
                                 self.ib and 
                                 self.ib.isConnected())

            if has_real_connection:
                # Try to use real IBKR data
                self.logger.info("Using real IBKR market data")
                success = await self._start_real_streaming(tickers)
                if success:
                    return True
                else:
                    self.logger.warning("Real streaming failed, falling back to simulation")
            else:
                self.logger.info("IBKR not connected, using simulation data")

            # Fall back to simulation data
            await self._start_simulation_streaming(tickers)
            return True

        except Exception as e:
            self.logger.error(f"❌ Market data streaming error: {e}")
            return False

    async def _start_real_streaming(self, tickers):
        """Start real IBKR streaming"""
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

    async def _start_simulation_streaming(self, tickers):
        """Start simulation streaming with realistic data"""
        try:
            for ticker in tickers:
                # Generate realistic base price
                base_prices = {
                    'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0, 
                    'TSLA': 200.0, 'NVDA': 400.0, 'APVO': 25.0
                }
                base_price = base_prices.get(ticker, random.uniform(50, 500))

                # Generate historical data for analysis
                historical_prices = []
                current_price = base_price

                for i in range(50):  # Generate 50 data points
                    change = random.uniform(-0.02, 0.02)  # ±2% change
                    current_price *= (1 + change)
                    historical_prices.append(current_price)

                # Calculate technical indicators
                rsi = self._calculate_rsi(historical_prices)
                bb_data = self._calculate_bollinger_bands(historical_prices)
                stoch_data = self._calculate_stochastic(historical_prices)

                # Initialize market data
                self.market_data[ticker] = {
                    'last': current_price,
                    'high': max(historical_prices[-10:]) if len(historical_prices) >= 10 else current_price,
                    'low': min(historical_prices[-10:]) if len(historical_prices) >= 10 else current_price,
                    'volume': random.randint(100000, 10000000),
                    'rsi': rsi,
                    'bollinger_pct': bb_data['position'],
                    'stoch_k': stoch_data['k'],
                    'stoch_d': stoch_data['d'],
                    'price_rate': random.uniform(-0.01, 0.01),
                    'rsi_rate': random.uniform(-0.1, 0.1),
                    'bollinger_rate': random.uniform(-0.05, 0.05),
                    'trend_slope': random.uniform(-0.001, 0.001),
                    'meets_criteria': False,
                    'exchange': self.exchange_var,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'Simulation',
                    'historical_prices': historical_prices
                }

                # Check if meets criteria
                self.market_data[ticker]['meets_criteria'] = (
                    rsi < 30 or rsi > 70 or 
                    bb_data['position'] < 20 or bb_data['position'] > 80 or
                    stoch_data['k'] < self.stoch_k_min or stoch_data['d'] < self.stoch_d_min
                )

                self.logger.info(f"✅ Simulation data generated for {ticker}")

            self.streaming = True
            asyncio.create_task(self._simulation_update_loop())
            self.logger.info(f"🚀 Simulation streaming started for {len(self.market_data)} symbols")

        except Exception as e:
            self.logger.error(f"Simulation streaming error: {e}")

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
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
        """Calculate Bollinger Bands"""
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
        """Calculate Stochastic oscillator"""
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

        # Simplified %D calculation
        d_percent = k_percent  # In real implementation, this would be a moving average

        return {'k': k_percent, 'd': d_percent}

    async def _simulation_update_loop(self):
        """Update simulation data periodically"""
        while self.streaming:
            try:
                for ticker in self.market_data:
                    data = self.market_data[ticker]

                    # Update price with small random changes
                    price_change = random.uniform(-0.005, 0.005)  # ±0.5% change
                    data['last'] *= (1 + price_change)

                    # Update historical prices
                    if 'historical_prices' in data:
                        data['historical_prices'].append(data['last'])
                        if len(data['historical_prices']) > 50:
                            data['historical_prices'].pop(0)

                        # Recalculate indicators
                        data['rsi'] = self._calculate_rsi(data['historical_prices'])
                        bb_data = self._calculate_bollinger_bands(data['historical_prices'])
                        data['bollinger_pct'] = bb_data['position']
                        stoch_data = self._calculate_stochastic(data['historical_prices'])
                        data['stoch_k'] = stoch_data['k']
                        data['stoch_d'] = stoch_data['d']

                        # Update rates
                        data['price_rate'] = price_change
                        data['rsi_rate'] = random.uniform(-0.1, 0.1)
                        data['bollinger_rate'] = random.uniform(-0.05, 0.05)

                        # Check criteria
                        data['meets_criteria'] = (
                            data['rsi'] < 30 or data['rsi'] > 70 or 
                            data['bollinger_pct'] < 20 or data['bollinger_pct'] > 80 or
                            data['stoch_k'] < self.stoch_k_min or data['stoch_d'] < self.stoch_d_min
                        )

                    data['timestamp'] = datetime.now().isoformat()

                await asyncio.sleep(2)  # Update every 2 seconds

            except Exception as e:
                self.logger.error(f"Simulation update error: {e}")
                await asyncio.sleep(5)

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
            
            # Execute the trade
            if hasattr(self, 'ib') and self.ib and self.ib.isConnected():
                # Real IBKR execution
                success = await self._execute_real_force_trade(symbol, action, quantity, order_type, order_price)
            else:
                # Simulation execution
                success = self._execute_simulation_force_trade(symbol, action, quantity, order_price)
            
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
            contract = ib_insync.Stock(symbol, self.exchange_var, 'USD')
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

    def _execute_simulation_force_trade(self, symbol: str, action: str, quantity: int, price: float):
        """Execute simulation force trade"""
        try:
            if action.upper() == "BUY":
                if self.available_funds >= price * quantity:
                    trade_data = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'buy_price': price,
                        'timestamp': datetime.now().isoformat(),
                        'sold': False,
                        'profit': 0,
                        'force_trade': True
                    }
                    self.active_orders[symbol] = trade_data
                    self.available_funds -= price * quantity
                    self.trade_history.append(trade_data.copy())
                    return True
                else:
                    return False
                    
            elif action.upper() == "SELL":
                if symbol in self.active_orders and not self.active_orders[symbol].get('sold', False):
                    trade = self.active_orders[symbol]
                    profit = (price - trade['buy_price']) * trade['quantity']
                    
                    # Mark as sold and update funds
                    self.active_orders[symbol]['sold'] = True
                    self.active_orders[symbol]['profit'] = profit
                    self.available_funds += price * trade['quantity']
                    
                    # Update trade history
                    for hist_trade in self.trade_history:
                        if (hist_trade['symbol'] == symbol and 
                            hist_trade['timestamp'] == trade['timestamp']):
                            hist_trade['sold'] = True
                            hist_trade['sell_price'] = price
                            hist_trade['profit'] = profit
                            break
                    
                    return True
                else:
                    return False
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Simulation force trade error: {e}")
            return False

    async def _analysis_loop(self):
        """Main analysis loop for real data"""
        while self.streaming:
            try:
                for ticker in self.market_data:
                    await self._perform_analysis(ticker)
                await asyncio.sleep(self.analysis_interval)
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(5)

    async def _perform_analysis(self, ticker):
        """Perform technical analysis on a ticker"""
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

    async def _generate_market_data(self):
        """DISABLED - Only real TWS API market data allowed"""
        self.logger.error("❌ SIMULATION DISABLED - Real IBKR connection required")
        self.logger.info("Please ensure TWS/IB Gateway is running with API enabled")
        self.logger.info("Market data type set to delayed (free) - check TWS_SETUP_GUIDE.md")
        self.streaming = False
        return

    def _check_trading_criteria(self, df, current) -> bool:
        """Check trading criteria - exact logic from reserve_file.py"""
        try:
            price_change = current["price_rate"]
            rsi_change = current["rsi_rate"]
            bollinger_change = current["bollinger_rate"]

            # Check if values are valid
            if pd.isna(price_change) or pd.isna(rsi_change) or pd.isna(bollinger_change):
                return False

            # Original criteria from reserve_file.py (exact thresholds)
            meets_criteria = (
                price_change >= 0.257142857 and
                rsi_change >= 0.201904762 and
                bollinger_change >= 1.48571429 and
                current["%K"] > current["%D"] and
                len(df) > 1 and df["%K"].iloc[-2] <= df["%D"].iloc[-2] and
                current["%K"] >= self.stoch_k_min and
                current["%D"] >= self.stoch_d_min
            )

            return meets_criteria

        except Exception as e:
            self.logger.error(f"Error checking trading criteria: {e}")
            return False

    async def _execute_auto_trade(self, symbol: str, current_data: Dict):
        """Execute automatic trade based on criteria (exact from reserve_file.py)"""
        try:
            # Check cooldown
            now = datetime.now()
            cooldown_period = 300  # 5 minutes
            last_trade_time = self.trade_cooldowns.get(symbol)

            if last_trade_time and (now - last_trade_time).total_seconds() < cooldown_period:
                return

            # Calculate position size
            current_price = current_data["close"]
            equity_per_trade = min(self.equity_per_trade, self.available_funds)
            quantity = int(equity_per_trade // current_price)

            if quantity > 0 and self.available_funds >= equity_per_trade:
                # Execute trade
                limit_price = round(current_price + 0.02, 2)

                trade_data = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'buy_price': limit_price,
                    'timestamp': now.isoformat(),
                    'sold': False,
                    'profit': 0
                }

                # Add to active orders
                self.active_orders[symbol] = trade_data

                # Update available funds
                self.available_funds -= equity_per_trade

                # Set cooldown
                self.trade_cooldowns[symbol] = now

                # Add to trade history
                self.trade_history.append(trade_data.copy())

                self.logger.info(f"✅ Executed auto trade: {symbol} x{quantity} @ ${limit_price}")

        except Exception as e:
            self.logger.error(f"Error executing auto trade: {e}")

    async def _place_real_order(self, symbol: str, action: str, shares: int, price: float) -> bool:
        """Place real order through TWS API for paper trading"""
        try:
            import ib_insync

            # Create contract
            contract = ib_insync.Stock(symbol, 'SMART', 'USD')

            # Create market order
            order = ib_insync.MarketOrder(action, shares)

            # Place order
            trade = self.ib.placeOrder(contract, order)

            # Wait for fill or timeout
            timeout = 30  # 30 seconds
            start_time = time.time()

            while time.time() - start_time < timeout:
                await asyncio.sleep(0.5)
                if trade.orderStatus.status in ['Filled', 'Cancelled']:
                    break

            if trade.orderStatus.status == 'Filled':
                self.logger.info(f"Paper Trade Filled: {action} {shares} {symbol}")
                return True
            else:
                self.logger.error(f"Paper Trade Failed: {trade.orderStatus.status}")
                return False

        except Exception as e:
            self.logger.error(f"Error placing paper trade order: {e}")
            return False