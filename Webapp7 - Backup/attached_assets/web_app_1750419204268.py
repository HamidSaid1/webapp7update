"""
Web-based Trading Application - Flask Interface
Provides web access to the sophisticated trading functionality
"""

import asyncio
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from threading import Thread
import nest_asyncio

from config import TradingConfig
from advanced_trading_engine import AdvancedTradingEngine
from logger_config import setup_logging

# Enable nested asyncio for Flask integration
nest_asyncio.apply()

class TradingWebApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.config = TradingConfig()
        self.loop = None  # Initialize loop attribute
        
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize trading components
        self.trading_engine = AdvancedTradingEngine(self.config)
        
        # Setup routes
        self.setup_routes()
        
        # Start asyncio loop in background
        self.setup_async_loop()
        
    def setup_async_loop(self):
        """Setup asyncio event loop in background thread"""
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            loop.run_forever()
        
        self.async_thread = Thread(target=run_loop, daemon=True)
        self.async_thread.start()
        
        # Wait a moment for loop to be ready
        import time
        time.sleep(0.1)
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            return render_template('dashboard.html', 
                                 config=self.config,
                                 status=self.get_system_status())
        
        @self.app.route('/api/status')
        def api_status():
            """Get system status"""
            return jsonify(self.get_system_status())
        
        @self.app.route('/api/market_data')
        def api_market_data():
            """Get current market data"""
            return jsonify(self.trading_engine.get_market_data())
        
        @self.app.route('/api/positions')
        def api_positions():
            """Get current positions"""
            return jsonify(self.trading_engine.get_positions())
        
        @self.app.route('/api/trade_history')
        def api_trade_history():
            """Get trade history"""
            return jsonify(self.trading_engine.get_trade_history())
        
        @self.app.route('/api/connect', methods=['POST'])
        def api_connect():
            """Connect to IBKR"""
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.trading_engine.connect(), self.loop
                )
                result = future.result(timeout=30)
                return jsonify({'success': True, 'connected': result})
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/disconnect', methods=['POST'])
        def api_disconnect():
            """Disconnect from IBKR"""
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.trading_engine.disconnect(), self.loop
                )
                future.result(timeout=10)
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"Disconnection error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/start_streaming', methods=['POST'])
        def api_start_streaming():
            """Start market data streaming"""
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.trading_engine.start_market_data(), self.loop
                )
                future.result(timeout=10)
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"Streaming start error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/stop_streaming', methods=['POST'])
        def api_stop_streaming():
            """Stop market data streaming"""
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.trading_engine.stop_market_data(), self.loop
                )
                future.result(timeout=10)
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"Streaming stop error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/update_config', methods=['POST'])
        def api_update_config():
            """Update trading configuration"""
            try:
                config_data = request.get_json()
                
                # Update configuration
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                # Validate configuration
                self.config.validate_config()
                
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"Config update error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/force_trade', methods=['POST'])
        def api_force_trade():
            """Execute force trade"""
            try:
                trade_data = request.get_json()
                symbol = trade_data.get('symbol')
                action = trade_data.get('action', 'BUY')
                
                if not symbol:
                    return jsonify({'success': False, 'error': 'Symbol required'})
                
                future = asyncio.run_coroutine_threadsafe(
                    self.trading_engine.force_trade(symbol, action), self.loop
                )
                result = future.result(timeout=30)
                
                return jsonify({'success': True, 'trade_result': result})
            except Exception as e:
                self.logger.error(f"Force trade error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/close_position', methods=['POST'])
        def api_close_position():
            """Close specific position"""
            try:
                position_data = request.get_json()
                symbol = position_data.get('symbol')
                
                if not symbol:
                    return jsonify({'success': False, 'error': 'Symbol required'})
                
                future = asyncio.run_coroutine_threadsafe(
                    self.trading_engine.close_position(symbol), self.loop
                )
                result = future.result(timeout=30)
                
                return jsonify({'success': True, 'close_result': result})
            except Exception as e:
                self.logger.error(f"Close position error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/analytics')
        def api_analytics():
            """Get performance analytics"""
            return jsonify(self.trading_engine.get_analytics())
        
        @self.app.route('/api/parameters', methods=['GET'])
        def get_parameters():
            """Get current trading parameters"""
            try:
                if hasattr(self.trading_engine, 'get_parameters'):
                    parameters = self.trading_engine.get_parameters()
                else:
                    parameters = {}
                return jsonify(parameters)
            except Exception as e:
                self.logger.error(f"Get parameters error: {e}")
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/parameters', methods=['POST'])
        def update_parameters():
            """Update trading parameters"""
            try:
                data = request.get_json()
                if hasattr(self.trading_engine, 'update_parameters'):
                    self.trading_engine.update_parameters(data)
                return jsonify({'success': True, 'message': 'Parameters updated successfully'})
            except Exception as e:
                self.logger.error(f"Update parameters error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/settings')
        def settings():
            """Enhanced settings page with authentic parameters"""
            return render_template('enhanced_settings.html', config=self.config)
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics page"""
            return render_template('analytics.html', 
                                 analytics=self.trading_engine.get_analytics())
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'connected': self.trading_engine.is_connected(),
            'streaming': self.trading_engine.is_streaming(),
            'force_trade_mode': getattr(self.trading_engine, 'force_trade_mode', False),
            'total_positions': len(self.trading_engine.get_positions()),
            'daily_pnl': self.trading_engine.get_daily_pnl(),
            'account_balance': self.trading_engine.get_account_balance(),
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.logger.info(f"Starting AutoTrade Plus Web Interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    web_app = TradingWebApp()
    web_app.run()