"""
AutoTrade Plus - Sophisticated Stock Trading Platform
Exact implementation of reserve_file.py calculations with IBKR integration
"""

import os
import logging
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import asyncio
import nest_asyncio

from config import TradingConfig
from trading_engine import AdvancedTradingEngine

# Apply nest_asyncio for compatibility
nest_asyncio.apply()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_trading_secret_key")

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize trading components
config = TradingConfig()
trading_engine = AdvancedTradingEngine(config)

@app.route('/')
def dashboard():
    """Main trading dashboard"""
    return render_template('dashboard.html')

@app.route('/settings')
def settings():
    """Trading settings and configuration"""
    return render_template('settings.html')

@app.route('/analytics')
def analytics():
    """Trading analytics and performance"""
    return render_template('analytics.html')

# API Routes
@app.route('/api/status')
def api_status():
    """Get system connection status"""
    return jsonify({
        'connected': trading_engine.is_connected(),
        'streaming': trading_engine.is_streaming(),
        'force_trade_mode': trading_engine.force_trade_enabled,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/market_data')
def api_market_data():
    """Get real-time market data with authentic calculations"""
    try:
        # Convert market data to JSON-serializable format
        json_data = {}
        for symbol, data in trading_engine.market_data.items():
            json_data[symbol] = {
                'last': float(data.get('last', 0)),
                'high': float(data.get('high', 0)),
                'low': float(data.get('low', 0)),
                'volume': int(data.get('volume', 0)),
                'rsi': float(data.get('rsi', 0)),
                'bollinger_pct': float(data.get('bollinger_pct', 0)),
                'stoch_k': float(data.get('stoch_k', 0)),
                'stoch_d': float(data.get('stoch_d', 0)),
                'price_rate': float(data.get('price_rate', 0)),
                'rsi_rate': float(data.get('rsi_rate', 0)),
                'bollinger_rate': float(data.get('bollinger_rate', 0)),
                'trend_slope': float(data.get('trend_slope', 0)),
                'meets_criteria': bool(data.get('meets_criteria', False)),
                'exchange': str(data.get('exchange', 'SMART')),
                'timestamp': str(data.get('timestamp', '')),
                'status': str(data.get('status', ''))
            }
        return jsonify(json_data)
    except Exception as e:
        app.logger.error(f"Error in api_market_data: {e}")
        return jsonify({})

@app.route('/api/analytics')
def api_analytics():
    """Get trading analytics and metrics"""
    total_trades = len(trading_engine.trade_history)
    total_profit = sum(trade.get('profit', 0) for trade in trading_engine.trade_history)
    win_rate = 0.0
    
    if total_trades > 0:
        winning_trades = sum(1 for trade in trading_engine.trade_history if trade.get('profit', 0) > 0)
        win_rate = (winning_trades / total_trades) * 100
    
    return jsonify({
        'total_trades': total_trades,
        'total_profit': total_profit,
        'win_rate': win_rate,
        'available_funds': trading_engine.available_funds,
        'active_positions': len(trading_engine.active_orders),
        'last_price_change': trading_engine.last_price_change,
        'last_rsi_change': trading_engine.last_rsi_change,
        'last_bollinger_change': trading_engine.last_bollinger_change,
        'equity_per_trade': trading_engine.equity_per_trade,
        'rsi_period': trading_engine.rsi_period,
        'bb_period': trading_engine.bb_period,
        'stoch_k_period': trading_engine.stoch_k_period,
        'stoch_d_period': trading_engine.stoch_d_period,
        'stoch_k_min': trading_engine.stoch_k_min,
        'stoch_d_min': trading_engine.stoch_d_min
    })

@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """Get current trading parameters"""
    return jsonify({
        'tickers': trading_engine.tickers,
        'data_points': trading_engine.data_points,
        'chart_interval': trading_engine.chart_interval,
        'analysis_interval': trading_engine.analysis_interval,
        'rsi_period': trading_engine.rsi_period,
        'bb_period': trading_engine.bb_period,
        'stoch_k_period': trading_engine.stoch_k_period,
        'stoch_d_period': trading_engine.stoch_d_period,
        'stoch_k_min': trading_engine.stoch_k_min,
        'stoch_d_min': trading_engine.stoch_d_min,
        'duration_str': trading_engine.duration_str,
        'equity_per_trade': trading_engine.equity_per_trade,
        'exchange_var': trading_engine.exchange_var,
        'hard_stop_loss': trading_engine.hard_stop_loss
    })

@app.route('/api/parameters', methods=['POST'])
def update_parameters():
    """Update trading parameters"""
    try:
        data = request.get_json()
        for key, value in data.items():
            if hasattr(trading_engine, key):
                setattr(trading_engine, key, value)
        
        logger.info(f"Updated parameters: {data}")
        return jsonify({'success': True, 'message': 'Parameters updated successfully'})
    except Exception as e:
        logger.error(f"Error updating parameters: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/connect', methods=['POST'])
def api_connect():
    """Connect to trading system"""
    try:
        success = asyncio.run(trading_engine.connect())
        if success:
            return jsonify({'success': True, 'message': 'Connected to IBKR API'})
        else:
            return jsonify({'success': False, 'error': 'Connection failed'})
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    """Disconnect from trading system"""
    try:
        asyncio.run(trading_engine.disconnect())
        return jsonify({'success': True, 'message': 'Disconnected from IBKR API'})
    except Exception as e:
        logger.error(f"Disconnection error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_streaming', methods=['POST'])
def api_start_streaming():
    """Start market data streaming"""
    try:
        success = asyncio.run(trading_engine.start_market_data())
        if success:
            return jsonify({'success': True, 'message': 'Market data streaming started'})
        else:
            return jsonify({'success': False, 'error': 'Failed to start market data streaming'})
    except Exception as e:
        logger.error(f"Streaming start error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_streaming', methods=['POST'])
def api_stop_streaming():
    """Stop market data streaming"""
    try:
        asyncio.run(trading_engine.stop_market_data())
        return jsonify({'success': True, 'message': 'Market data streaming stopped'})
    except Exception as e:
        logger.error(f"Streaming stop error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/toggle_force_trade', methods=['POST'])
def api_toggle_force_trade():
    """Toggle force trade mode"""
    try:
        trading_engine.force_trade_enabled = not trading_engine.force_trade_enabled
        status = "enabled" if trading_engine.force_trade_enabled else "disabled"
        return jsonify({
            'success': True, 
            'message': f'Force trade mode {status}',
            'force_trade_enabled': trading_engine.force_trade_enabled
        })
    except Exception as e:
        logger.error(f"Force trade toggle error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/force_trade', methods=['POST'])
def api_force_trade():
    """Execute a force trade"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        action = data.get('action', '').strip().upper()
        
        if not symbol or not action:
            return jsonify({'success': False, 'error': 'Symbol and action are required'})
        
        if action not in ['BUY', 'SELL']:
            return jsonify({'success': False, 'error': 'Action must be BUY or SELL'})
        
        if not trading_engine.force_trade_enabled:
            return jsonify({'success': False, 'error': 'Force trade mode is not enabled'})
        
        # Execute force trade
        result = asyncio.run(trading_engine.execute_force_trade(symbol, action))
        
        if 'success' in result:
            return jsonify({'success': True, 'message': result['success'], 'details': result.get('details', {})})
        else:
            return jsonify({'success': False, 'error': result.get('error', 'Unknown error')})
            
    except Exception as e:
        logger.error(f"Force trade API error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trade_history')
def api_trade_history():
    """Get trade history"""
    return jsonify(trading_engine.trade_history)

@app.route('/api/active_orders')
def api_active_orders():
    """Get active orders"""
    return jsonify(trading_engine.active_orders)

if __name__ == '__main__':
    print("🚀 Starting AutoTrade Plus - Sophisticated Trading Platform")
    print("📊 Exact reserve_file.py calculations integrated")
    print("🔗 IBKR API integration ready")
    print("💼 Professional trading interface loaded")
    app.run(host='0.0.0.0', port=5000, debug=True)
