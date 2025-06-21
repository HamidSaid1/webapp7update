"""
Advanced Risk Management System
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

class RiskManager:
    """Advanced risk management with comprehensive controls"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        
        # Position tracking
        self.open_positions = {}
        self.position_sizes = {}
        
        # Risk limits
        self.risk_limits = {
            'max_daily_loss': config.max_daily_loss,
            'max_daily_trades': config.max_daily_trades,
            'max_position_size': config.max_position_size,
            'max_total_exposure': config.max_position_size * config.max_open_trades
        }
        
    def check_daily_limits(self, current_trades: int, current_pnl: float) -> bool:
        """Check if daily trading limits are exceeded"""
        try:
            # Check daily loss limit
            if current_pnl <= -self.config.max_daily_loss:
                self.logger.warning(f"Daily loss limit exceeded: ${current_pnl:.2f}")
                return False
                
            # Check daily trade limit
            if current_trades >= self.config.max_daily_trades:
                self.logger.warning(f"Daily trade limit exceeded: {current_trades}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Daily limits check error: {e}")
            return False
            
    def calculate_position_size(self, symbol: str, price: float, account_balance: float) -> int:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Base position size on percentage of account
            base_size = account_balance * self.config.position_size_percent
            
            # Calculate shares based on price
            shares = int(base_size / price) if price > 0 else 0
            
            # Apply maximum position size limit
            max_shares = int(self.config.max_position_size / price) if price > 0 else 0
            shares = min(shares, max_shares)
            
            # Adjust for risk per trade
            risk_per_share = price * self.config.hard_stop_loss
            max_shares_by_risk = int(self.config.equity_per_trade / risk_per_share) if risk_per_share > 0 else shares
            shares = min(shares, max_shares_by_risk)
            
            # Ensure minimum viable position
            if shares < 1:
                shares = 0
                
            self.logger.debug(f"Position size for {symbol}: {shares} shares @ ${price:.2f}")
            
            return shares
            
        except Exception as e:
            self.logger.error(f"Position size calculation error for {symbol}: {e}")
            return 0
            
    def check_position_limits(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if position exceeds limits"""
        try:
            position_value = quantity * price
            
            # Check individual position size
            if position_value > self.config.max_position_size:
                self.logger.warning(f"Position size limit exceeded for {symbol}: ${position_value:.2f}")
                return False
                
            # Check total exposure
            current_exposure = sum(
                pos['quantity'] * pos['current_price'] 
                for pos in self.open_positions.values()
            )
            
            if current_exposure + position_value > self.config.max_position_size * self.config.max_open_trades:
                self.logger.warning(f"Total exposure limit exceeded: ${current_exposure + position_value:.2f}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Position limits check error for {symbol}: {e}")
            return False
            
    def calculate_stop_loss(self, symbol: str, entry_price: float, action: str) -> float:
        """Calculate stop loss price"""
        try:
            if action == 'BUY':
                stop_loss = entry_price * (1 - self.config.hard_stop_loss)
            else:  # SELL
                stop_loss = entry_price * (1 + self.config.hard_stop_loss)
                
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Stop loss calculation error for {symbol}: {e}")
            return entry_price
            
    def calculate_take_profit(self, symbol: str, entry_price: float, action: str) -> float:
        """Calculate take profit price"""
        try:
            if action == 'BUY':
                take_profit = entry_price * (1 + self.config.take_profit)
            else:  # SELL
                take_profit = entry_price * (1 - self.config.take_profit)
                
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Take profit calculation error for {symbol}: {e}")
            return entry_price
            
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        """Calculate risk-to-reward ratio"""
        try:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk > 0:
                return reward / risk
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Risk-reward ratio calculation error: {e}")
            return 0.0
            
    def update_position_risk(self, symbol: str, current_price: float):
        """Update risk metrics for open position"""
        try:
            if symbol not in self.open_positions:
                return
                
            position = self.open_positions[symbol]
            entry_price = position['entry_price']
            quantity = position['quantity']
            action = position['action']
            
            # Calculate current P&L
            if action == 'BUY':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:  # SELL
                unrealized_pnl = (entry_price - current_price) * quantity
                
            # Update position
            position.update({
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'last_update': datetime.now()
            })
            
            # Check if stop loss should be triggered
            if self.should_trigger_stop_loss(symbol, current_price):
                self.logger.warning(f"Stop loss triggered for {symbol} at ${current_price:.2f}")
                
        except Exception as e:
            self.logger.error(f"Position risk update error for {symbol}: {e}")
            
    def should_trigger_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        try:
            if symbol not in self.open_positions:
                return False
                
            position = self.open_positions[symbol]
            stop_loss = position.get('stop_loss', 0)
            action = position['action']
            
            if action == 'BUY' and current_price <= stop_loss:
                return True
            elif action == 'SELL' and current_price >= stop_loss:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Stop loss check error for {symbol}: {e}")
            return False
            
    def should_trigger_take_profit(self, symbol: str, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        try:
            if symbol not in self.open_positions:
                return False
                
            position = self.open_positions[symbol]
            take_profit = position.get('take_profit', 0)
            action = position['action']
            
            if action == 'BUY' and current_price >= take_profit:
                return True
            elif action == 'SELL' and current_price <= take_profit:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Take profit check error for {symbol}: {e}")
            return False
            
    def calculate_portfolio_risk(self, account_balance: float) -> Dict[str, float]:
        """Calculate overall portfolio risk metrics"""
        try:
            total_exposure = 0.0
            total_unrealized_pnl = 0.0
            
            for position in self.open_positions.values():
                position_value = position['quantity'] * position.get('current_price', position['entry_price'])
                total_exposure += position_value
                total_unrealized_pnl += position.get('unrealized_pnl', 0)
                
            # Calculate risk metrics
            exposure_ratio = total_exposure / account_balance if account_balance > 0 else 0
            
            # Update max drawdown
            current_balance = account_balance + total_unrealized_pnl
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
                
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance if self.peak_balance > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            return {
                'total_exposure': total_exposure,
                'exposure_ratio': exposure_ratio,
                'unrealized_pnl': total_unrealized_pnl,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': current_drawdown,
                'risk_level': self.assess_risk_level(exposure_ratio, current_drawdown)
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio risk calculation error: {e}")
            return {
                'total_exposure': 0.0,
                'exposure_ratio': 0.0,
                'unrealized_pnl': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'risk_level': 'LOW'
            }
            
    def assess_risk_level(self, exposure_ratio: float, drawdown: float) -> str:
        """Assess overall risk level"""
        try:
            if exposure_ratio > 0.8 or drawdown > 0.15:
                return 'HIGH'
            elif exposure_ratio > 0.5 or drawdown > 0.08:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            self.logger.error(f"Risk level assessment error: {e}")
            return 'LOW'
            
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            return {
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'max_drawdown': self.max_drawdown,
                'open_positions': len(self.open_positions),
                'risk_limits': self.risk_limits,
                'limits_status': {
                    'daily_loss_ok': self.daily_pnl > -self.config.max_daily_loss,
                    'daily_trades_ok': self.daily_trades < self.config.max_daily_trades,
                    'positions_ok': len(self.open_positions) < self.config.max_open_trades
                }
            }
            
        except Exception as e:
            self.logger.error(f"Risk summary error: {e}")
            return {}
            
    def update_daily_stats(self, pnl: float, trades: int):
        """Update daily statistics"""
        try:
            self.daily_pnl = pnl
            self.daily_trades = trades
            
        except Exception as e:
            self.logger.error(f"Daily stats update error: {e}")
            
    def add_position(self, symbol: str, position_data: Dict):
        """Add new position to tracking"""
        try:
            self.open_positions[symbol] = position_data
            self.logger.info(f"Added position tracking for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Add position error for {symbol}: {e}")
            
    def remove_position(self, symbol: str) -> Optional[Dict]:
        """Remove position from tracking"""
        try:
            if symbol in self.open_positions:
                position = self.open_positions.pop(symbol)
                self.logger.info(f"Removed position tracking for {symbol}")
                return position
            return None
            
        except Exception as e:
            self.logger.error(f"Remove position error for {symbol}: {e}")
            return None
