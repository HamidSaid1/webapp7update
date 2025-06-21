"""
Enhanced Web Application with Authentic Calculations from Reserve File
"""

from flask import Flask, render_template, jsonify, request
import logging
from datetime import datetime
import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import time

from config import TradingConfig
from logger_config import setup_logging

class EnhancedTradingWebApp:
    """Enhanced web application with authentic calculations from reserve_file.py"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.config = TradingConfig()
        
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Trading parameters from original reserve_file.py
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
        
        # Trading state
        self.connected = False
        self.streaming = False
        self.market_data = {}
        self.active_orders = {}
        self.trade_history = []
        self.available_funds = 1000.0
        self.force_trade_enabled = False
        
        # Analytics tracking
        self.last_price_change = "N/A"
        self.last_rsi_change = "N/A"
        self.last_bollinger_change = "N/A"
        
        # Setup routes
        self.setup_routes()
        
        # Start generating sample data
        self.generate_sample_data()
    
    def compute_slope(self, prices):
        """Compute slope of price trend using linear regression - from original reserve_file.py"""
        if len(prices) < 2:
            return 0.0
        x = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]
    
    def compute_rsi(self, df, period=14):
        """Compute RSI - exact implementation from reserve_file.py"""
        if len(df) < period + 1:
            return pd.Series([50.0] * len(df), index=df.index)
            
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
    
    def compute_stochastic(self, df, k_period=14, d_period=3):
        """Compute Stochastic - exact implementation from reserve_file.py"""
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
        """Compute Bollinger % - exact implementation from reserve_file.py"""
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
        """Convert chart interval to minutes - from reserve_file.py"""
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
    
    def check_trading_criteria(self, df, current) -> bool:
        """Check trading criteria - exact logic from reserve_file.py"""
        try:
            price_change = current["price_rate"]
            rsi_change = current["rsi_rate"]
            bollinger_change = current["bollinger_rate"]
            
            if pd.isna(price_change) or pd.isna(rsi_change) or pd.isna(bollinger_change):
                return False
            
            # Original criteria from reserve_file.py
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
    
    def generate_sample_data(self):
        """Generate realistic market data with authentic calculations"""
        base_prices = {
            'AAPL': 150.0, 'MSFT': 280.0, 'GOOGL': 2500.0, 
            'TSLA': 200.0, 'NVDA': 450.0, 'ABTS': 25.0
        }
        
        for symbol in self.tickers.split(','):
            symbol = symbol.strip()
            if symbol in base_prices:
                # Generate historical data
                historical_data = []
                base_price = base_prices[symbol]
                
                for i in range(100):
                    price_change = random.uniform(-0.02, 0.02)
                    base_price *= (1 + price_change)
                    
                    historical_data.append({
                        'date': datetime.now(),
                        'close': base_price,
                        'high': base_price * 1.005,
                        'low': base_price * 0.995,
                        'volume': random.randint(1000, 10000)
                    })
                
                # Convert to DataFrame and apply authentic calculations
                df = pd.DataFrame(historical_data)
                df["RSI"] = self.compute_rsi(df, period=self.rsi_period)
                df["Bollinger %"] = self.compute_bollinger(df, period=self.bb_period)
                df = self.compute_stochastic(df, k_period=self.stoch_k_period, d_period=self.stoch_d_period)
                
                # Calculate rates of change (authentic from reserve_file.py)
                window = self.analysis_interval
                minutes_per_bar = self.interval_to_minutes()
                
                df["price_pct"] = df["close"].pct_change(fill_method=None) * 100
                df["price_rate"] = (df["price_pct"] / minutes_per_bar).rolling(window=window).mean()
                df["rsi_pct"] = df["RSI"].pct_change(fill_method=None) * 100
                df["rsi_rate"] = (df["rsi_pct"] / minutes_per_bar).rolling(window=window).mean()
                df["bollinger_pct"] = df["Bollinger %"].pct_change(fill_method=None) * 100
                df["bollinger_rate"] = (df["bollinger_pct"] / minutes_per_bar).rolling(window=window).mean()
                
                current = df.iloc[-1]
                
                # Check trading criteria
                meets_criteria = self.check_trading_criteria(df, current)
                
                # Store market data
                self.market_data[symbol] = {
                    'last': current["close"],
                    'rsi': current["RSI"],
                    'bollinger_pct': current["Bollinger %"],
                    'stoch_k': current["%K"],
                    'stoch_d': current["%D"],
                    'price_rate': current["price_rate"],
                    'rsi_rate': current["rsi_rate"],
                    'bollinger_rate': current["bollinger_rate"],
                    'meets_criteria': meets_criteria,
                    'exchange': self.exchange_var,
                    'timestamp': datetime.now()
                }
                
                # Update analytics
                if not pd.isna(current["price_rate"]):
                    self.last_price_change = current["price_rate"]
                if not pd.isna(current["rsi_rate"]):
                    self.last_rsi_change = current["rsi_rate"]
                if not pd.isna(current["bollinger_rate"]):
                    self.last_bollinger_change = current["bollinger_rate"]
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard"""
            return render_template('dashboard.html')
        
        @self.app.route('/settings')
        def settings():
            """Enhanced settings page with authentic parameters"""
            return render_template('enhanced_settings.html')
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics page"""
            return render_template('analytics.html')
        
        @self.app.route('/api/status')
        def api_status():
            """Get system status"""
            return jsonify({
                'connected': self.connected,
                'streaming': self.streaming,
                'force_trade_mode': self.force_trade_enabled,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/market_data')
        def api_market_data():
            """Get market data with authentic calculations"""
            self.generate_sample_data()  # Refresh data
            return jsonify(self.market_data)
        
        @self.app.route('/api/analytics')
        def api_analytics():
            """Get analytics with authentic metrics"""
            total_trades = len(self.trade_history)
            total_profit = sum(trade.get('profit', 0) for trade in self.trade_history)
            win_rate = 0.0
            
            if total_trades > 0:
                winning_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
                win_rate = (winning_trades / total_trades) * 100
            
            return jsonify({
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
            })
        
        @self.app.route('/api/parameters', methods=['GET'])
        def get_parameters():
            """Get current trading parameters from original reserve_file.py"""
            return jsonify({
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
            })
        
        @self.app.route('/api/parameters', methods=['POST'])
        def update_parameters():
            """Update trading parameters"""
            try:
                data = request.get_json()
                
                # Update parameters
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        self.logger.info(f"Updated parameter {key} to {value}")
                
                # Regenerate data with new parameters
                self.generate_sample_data()
                
                return jsonify({'success': True, 'message': 'Parameters updated successfully'})
            except Exception as e:
                self.logger.error(f"Update parameters error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/connect', methods=['POST'])
        def api_connect():
            """Simulate connection"""
            self.connected = True
            return jsonify({'success': True, 'message': 'Connected to trading system'})
        
        @self.app.route('/api/disconnect', methods=['POST'])
        def api_disconnect():
            """Simulate disconnection"""
            self.connected = False
            self.streaming = False
            return jsonify({'success': True, 'message': 'Disconnected from trading system'})
        
        @self.app.route('/api/start_streaming', methods=['POST'])
        def api_start_streaming():
            """Start market data streaming"""
            if not self.connected:
                return jsonify({'success': False, 'error': 'Not connected'})
            
            self.streaming = True
            return jsonify({'success': True, 'message': 'Market data streaming started'})
        
        @self.app.route('/api/stop_streaming', methods=['POST'])
        def api_stop_streaming():
            """Stop market data streaming"""
            self.streaming = False
            return jsonify({'success': True, 'message': 'Market data streaming stopped'})
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.logger.info(f"Starting Enhanced AutoTrade Plus Web Interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    web_app = EnhancedTradingWebApp()
    web_app.run()