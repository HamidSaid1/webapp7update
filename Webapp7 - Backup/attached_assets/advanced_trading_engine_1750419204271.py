"""
Advanced Trading Engine with Authentic Calculations from Reserve File
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
    """Advanced trading engine with authentic calculations from reserve_file.py"""
    
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
        
        # User configurable parameters from reserve_file.py
        self.tickers = "AAPL,MSFT,GOOGL,TSLA,NVDA"
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
        """Compute slope of price trend using linear regression"""
        if len(prices) < 2:
            return 0.0
        x = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]
    
    def calculate_adaptive_thresholds(self, spike_stats):
        """Calculate adaptive thresholds based on collected spike metrics"""
        if spike_stats["count"] < 3:
            return None
            
        return {
            "price_rate": sum(spike_stats["price_rates"]) / len(spike_stats["price_rates"]),
            "rsi_rate": sum(spike_stats["rsi_rates"]) / len(spike_stats["rsi_rates"]),
            "boll_rate": sum(spike_stats["boll_rates"]) / len(spike_stats["boll_rates"]),
            "slope": sum(spike_stats["slopes"]) / len(spike_stats["slopes"])
        }
    
    def compute_rsi(self, df, period=14):
        """Compute the Relative Strength Index (RSI) based on historical price data"""
        if len(df) < period + 1:
            return pd.Series([50.0] * len(df), index=df.index)
            
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
    
    def compute_stochastic(self, df, k_period=14, d_period=3):
        """Compute Stochastic Oscillator"""
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
        
        # Avoid division by zero
        range_val = high_max - low_min
        range_val = range_val.replace(0, 1)
        
        df['%K'] = ((df['close'] - low_min) / range_val * 100).fillna(50.0)
        df['%D'] = df['%K'].rolling(window=d_period).mean().fillna(50.0)
        return df
    
    def compute_bollinger(self, df, period=20, num_std=2):
        """Compute Bollinger % Position based on historical price data"""
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
        """Converts selected chart interval string to number of minutes"""
        mapping = {
            "5 sec": 5 / 60,
            "10 sec": 10 / 60,
            "30 sec": 30 / 60,
            "1 min": 1,
            "3 mins": 3,
            "5 mins": 5,
            "10 mins": 10,
            "15 mins": 15
        }
        return mapping.get(self.chart_interval, 1)
    
    async def connect(self) -> bool:
        """Simulate connection to broker"""
        await asyncio.sleep(0.5)
        self.connected = True
        self.logger.info("Advanced Trading Engine connected")
        return True
    
    async def disconnect(self):
        """Disconnect from broker"""
        self.connected = False
        self.streaming = False
        self.logger.info("Advanced Trading Engine disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connected
    
    def is_streaming(self) -> bool:
        """Check if streaming"""
        return self.streaming
    
    async def start_market_data(self):
        """Start market data streaming with authentic calculations"""
        if not self.connected:
            return
            
        self.streaming = True
        self.logger.info("Starting advanced market data streaming")
        
        # Start background task for market data updates
        asyncio.create_task(self._generate_advanced_market_data())
    
    async def stop_market_data(self):
        """Stop market data streaming"""
        self.streaming = False
        self.logger.info("Stopped advanced market data streaming")
    
    async def _generate_advanced_market_data(self):
        """Generate realistic market data with authentic technical analysis"""
        base_prices = {
            'AAPL': 150.0, 'MSFT': 280.0, 'GOOGL': 2500.0, 
            'TSLA': 200.0, 'NVDA': 450.0, 'ABTS': 25.0
        }
        
        # Generate historical data for each symbol
        for symbol in self.tickers.split(','):
            symbol = symbol.strip()
            if symbol in base_prices:
                # Generate 100 bars of historical data
                historical_data = []
                base_price = base_prices[symbol]
                
                for i in range(100):
                    # Create realistic price movement
                    price_change = random.uniform(-0.02, 0.02)
                    base_price *= (1 + price_change)
                    
                    historical_data.append({
                        'date': datetime.now() - timedelta(minutes=5*(100-i)),
                        'close': base_price,
                        'high': base_price * 1.005,
                        'low': base_price * 0.995,
                        'volume': random.randint(1000, 10000)
                    })
                
                # Convert to DataFrame for analysis
                df = pd.DataFrame(historical_data)
                
                # Apply authentic calculations from reserve_file.py
                df["RSI"] = self.compute_rsi(df, period=self.rsi_period)
                df["Bollinger %"] = self.compute_bollinger(df, period=self.bb_period)
                df = self.compute_stochastic(df, k_period=self.stoch_k_period, d_period=self.stoch_d_period)
                
                # Calculate rates of change (key part of original algorithm)
                window = self.analysis_interval
                minutes_per_bar = self.interval_to_minutes()
                
                df["price_pct"] = df["close"].pct_change(fill_method=None) * 100
                df["price_rate"] = (df["price_pct"] / minutes_per_bar).rolling(window=window).mean()
                df["rsi_pct"] = df["RSI"].pct_change(fill_method=None) * 100
                df["rsi_rate"] = (df["rsi_pct"] / minutes_per_bar).rolling(window=window).mean()
                df["bollinger_pct"] = df["Bollinger %"].pct_change(fill_method=None) * 100
                df["bollinger_rate"] = (df["bollinger_pct"] / minutes_per_bar).rolling(window=window).mean()
                
                # Get current values
                current = df.iloc[-1]
                
                # Calculate trend slope
                recent_prices = df["close"].tail(10).values
                trend_slope = self.compute_slope(recent_prices)
                
                # Store market data with all authentic calculations
                self.market_data[symbol] = {
                    'last': current["close"],
                    'high': current["high"],
                    'low': current["low"],
                    'volume': current["volume"],
                    'rsi': current["RSI"],
                    'bollinger_pct': current["Bollinger %"],
                    'stoch_k': current["%K"],
                    'stoch_d': current["%D"],
                    'price_rate': current["price_rate"],
                    'rsi_rate': current["rsi_rate"],
                    'bollinger_rate': current["bollinger_rate"],
                    'trend_slope': trend_slope,
                    'exchange': self.exchange_var,
                    'timestamp': datetime.now()
                }
                
                # Check trading criteria (from original reserve_file.py)
                meets_criteria = self._check_trading_criteria(df, current)
                
                # Auto-trade if criteria met or force trade enabled
                if (meets_criteria or self.force_trade_enabled) and symbol not in self.active_orders:
                    await self._execute_auto_trade(symbol, current)
        
        # Continue streaming while enabled
        if self.streaming:
            await asyncio.sleep(5)  # Update every 5 seconds
            await self._generate_advanced_market_data()
    
    def _check_trading_criteria(self, df, current) -> bool:
        """Check if trading criteria are met using original algorithm"""
        try:
            # Original criteria from reserve_file.py
            price_change = current["price_rate"]
            rsi_change = current["rsi_rate"]
            bollinger_change = current["bollinger_rate"]
            
            # Check if values are valid
            if pd.isna(price_change) or pd.isna(rsi_change) or pd.isna(bollinger_change):
                return False
            
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
        """Execute automatic trade based on criteria"""
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
                    'timestamp': now,
                    'sold': False
                }
                
                self.active_orders[symbol] = trade_data
                self.available_funds -= equity_per_trade
                self.trade_cooldowns[symbol] = now
                
                # Store analytics
                self.last_price_change = current_data["price_rate"]
                self.last_rsi_change = current_data["rsi_rate"]
                self.last_bollinger_change = current_data["bollinger_rate"]
                
                self.logger.info(f"Auto-executed trade: {symbol} x{quantity} @ ${limit_price}")
                
        except Exception as e:
            self.logger.error(f"Error executing auto trade for {symbol}: {e}")
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data with authentic calculations"""
        return self.market_data
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        positions = {}
        for symbol, trade in self.active_orders.items():
            if not trade.get('sold', False):
                current_price = self.market_data.get(symbol, {}).get('last', trade['buy_price'])
                unrealized_pnl = (current_price - trade['buy_price']) * trade['quantity']
                
                positions[symbol] = {
                    'quantity': trade['quantity'],
                    'avg_price': trade['buy_price'],
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'timestamp': trade['timestamp']
                }
        
        return positions
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history
    
    def get_daily_pnl(self) -> float:
        """Get daily P&L"""
        total_pnl = 0.0
        for position in self.get_positions().values():
            total_pnl += position['unrealized_pnl']
        return total_pnl
    
    def get_account_balance(self) -> float:
        """Get account balance"""
        return self.available_funds
    
    async def force_trade(self, symbol: str, action: str) -> Dict[str, Any]:
        """Execute a force trade"""
        if symbol not in self.market_data:
            return {'error': f'No market data for {symbol}'}
        
        current_price = self.market_data[symbol]['last']
        quantity = 10  # Default quantity for force trades
        
        if action.upper() == 'BUY':
            if self.available_funds >= current_price * quantity:
                trade_data = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'buy_price': current_price,
                    'timestamp': datetime.now(),
                    'sold': False,
                    'force_trade': True
                }
                
                self.active_orders[symbol] = trade_data
                self.available_funds -= current_price * quantity
                
                return {'success': f'Force bought {quantity} shares of {symbol} @ ${current_price:.2f}'}
            else:
                return {'error': 'Insufficient funds'}
        
        elif action.upper() == 'SELL':
            if symbol in self.active_orders and not self.active_orders[symbol].get('sold', False):
                trade = self.active_orders[symbol]
                profit = (current_price - trade['buy_price']) * trade['quantity']
                
                # Mark as sold and update funds
                self.active_orders[symbol]['sold'] = True
                self.available_funds += current_price * trade['quantity']
                
                # Add to trade history
                self.trade_history.append({
                    'symbol': symbol,
                    'buy_price': trade['buy_price'],
                    'sell_price': current_price,
                    'quantity': trade['quantity'],
                    'profit': profit,
                    'timestamp': datetime.now()
                })
                
                return {'success': f'Force sold {symbol} for ${profit:.2f} profit'}
            else:
                return {'error': f'No open position for {symbol}'}
    
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position"""
        result = await self.force_trade(symbol, 'SELL')
        return result if result else {'error': 'Failed to close position'}
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get performance analytics with authentic metrics"""
        total_trades = len(self.trade_history)
        total_profit = sum(trade['profit'] for trade in self.trade_history)
        win_rate = 0.0
        
        if total_trades > 0:
            winning_trades = sum(1 for trade in self.trade_history if trade['profit'] > 0)
            win_rate = (winning_trades / total_trades) * 100
        
        return {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'available_funds': self.available_funds,
            'daily_pnl': self.get_daily_pnl(),
            'active_positions': len([p for p in self.active_orders.values() if not p.get('sold', False)]),
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
    
    def update_parameters(self, params: Dict[str, Any]):
        """Update trading parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated parameter {key} to {value}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current trading parameters"""
        return {
            'tickers': self.tickers,
            'data_points': self.data_points,
            'chart_interval': self.chart_interval,
            'analysis_interval': self.analysis_interval,
            'rsi_period': self.rsi_period,
            'bb_period': self.bb_period,
            'stoch_k_period': self.stoch_k_period,
            'stoch_d_period': self.stoch_d_period,
            'stoch_k_min': self.stoch_k_min,
            'stoch_d_min': self.stoch_d_min,
            'duration_str': self.duration_str,
            'equity_per_trade': self.equity_per_trade,
            'exchange_var': self.exchange_var,
            'hard_stop_loss': self.hard_stop_loss
        }