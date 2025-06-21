"""
Simplified Trading Engine for Web Interface
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import random
import numpy as np

class SimplifiedTradingEngine:
    """Simplified trading engine for web interface demonstration"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.streaming = False
        self.positions = {}
        self.trade_history = []
        self.market_data = {}
        self.account_balance = 100000.0  # Starting balance
        self.daily_pnl = 0.0
        
    async def connect(self) -> bool:
        """Simulate connection to broker"""
        try:
            self.logger.info("Attempting to connect to broker...")
            await asyncio.sleep(1)  # Simulate connection time
            self.connected = True
            self.logger.info("Successfully connected to broker")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from broker"""
        self.connected = False
        self.streaming = False
        self.logger.info("Disconnected from broker")
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connected
    
    def is_streaming(self) -> bool:
        """Check if streaming"""
        return self.streaming
    
    async def start_market_data(self):
        """Start market data streaming"""
        if not self.connected:
            raise Exception("Not connected to broker")
        
        self.streaming = True
        self.logger.info("Market data streaming started")
        
        # Generate sample market data
        await self._generate_sample_data()
    
    async def stop_market_data(self):
        """Stop market data streaming"""
        self.streaming = False
        self.logger.info("Market data streaming stopped")
    
    async def _generate_sample_data(self):
        """Generate sample market data for demonstration"""
        symbols = self.config.default_tickers.split(',')
        
        for symbol in symbols:
            symbol = symbol.strip()
            # Generate realistic sample data
            base_price = random.uniform(50, 500)
            change = random.uniform(-5, 5)
            volume = random.randint(100000, 10000000)
            
            # Generate technical indicators
            rsi = random.uniform(20, 80)
            bb_position = random.uniform(0, 100)
            
            # Generate signal based on indicators
            if rsi < 30 and bb_position < 20:
                signal = "BUY"
            elif rsi > 70 and bb_position > 80:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            self.market_data[symbol] = {
                'price': base_price,
                'change': change,
                'volume': volume,
                'rsi': rsi,
                'bb_position': bb_position,
                'signal': signal,
                'timestamp': datetime.now()
            }
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data"""
        return self.market_data
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        return self.positions
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history
    
    def get_daily_pnl(self) -> float:
        """Get daily P&L"""
        return self.daily_pnl
    
    def get_account_balance(self) -> float:
        """Get account balance"""
        return self.account_balance
    
    async def force_trade(self, symbol: str, action: str) -> Dict[str, Any]:
        """Execute a force trade"""
        if not self.connected:
            raise Exception("Not connected to broker")
        
        try:
            # Get current market data for symbol
            if symbol not in self.market_data:
                await self._generate_sample_data()
            
            if symbol not in self.market_data:
                raise Exception(f"No market data for {symbol}")
            
            price = self.market_data[symbol]['price']
            quantity = int(self.config.equity_per_trade / price)
            
            if quantity <= 0:
                raise Exception("Invalid quantity calculated")
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now(),
                'status': 'FILLED'
            }
            
            # Update positions
            if symbol in self.positions:
                if action == 'BUY':
                    self.positions[symbol]['quantity'] += quantity
                else:  # SELL
                    self.positions[symbol]['quantity'] -= quantity
                    if self.positions[symbol]['quantity'] <= 0:
                        del self.positions[symbol]
            else:
                if action == 'BUY':
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_cost': price,
                        'unrealized_pnl': 0.0
                    }
            
            # Add to trade history
            self.trade_history.append(trade)
            
            # Update account balance
            if action == 'BUY':
                self.account_balance -= (quantity * price)
            else:
                self.account_balance += (quantity * price)
            
            self.logger.info(f"Trade executed: {action} {quantity} {symbol} @ ${price:.2f}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            raise
    
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position"""
        if symbol not in self.positions:
            raise Exception(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        quantity = position['quantity']
        
        # Execute sell order
        return await self.force_trade(symbol, 'SELL')
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get performance analytics"""
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        total_trades = len(self.trade_history)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        total_return = self.account_balance - 100000.0  # Initial balance
        
        analytics = {
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': random.uniform(0.5, 2.0),  # Sample value
            'max_drawdown': random.uniform(-5, -15),  # Sample value
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'largest_win': max((t.get('pnl', 0) for t in winning_trades), default=0),
            'largest_loss': min((t.get('pnl', 0) for t in losing_trades), default=0),
            'equity_curve': [{'equity': 100000 + i * 100} for i in range(total_trades)]
        }
        
        return analytics