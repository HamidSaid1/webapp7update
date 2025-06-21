"""
Enhanced Trading Engine with sophisticated calculations and adaptive algorithms
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, util
import nest_asyncio

from technical_analyzer import TechnicalAnalyzer
from risk_manager import RiskManager
from utils import (
    calculate_position_size, format_currency, format_percentage,
    validate_ticker_symbol, parse_ticker_list, generate_trade_id
)
from logger_config import log_trade_event, log_connection_event, log_risk_event

class TradingEngine:
    """Enhanced trading engine with sophisticated market analysis and execution"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # IBKR connection
        self.ib = IB()
        self.is_connected = False
        self.is_streaming = False
        
        # Analysis components
        self.technical_analyzer = TechnicalAnalyzer(config)
        self.risk_manager = RiskManager(config)
        
        # Data storage
        self.market_data = {}
        self.historical_data = {}
        self.open_positions = {}
        self.trade_history = []
        
        # Trading state
        self.force_trade_mode = False
        self.force_trade_cooldown = {}
        self.last_trade_time = {}
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.account_balance = 0.0
        
        # Callbacks
        self.gui_callback = None
        self.trade_callback = None
        self.status_callback = None
        
        # Event loop
        self.event_loop = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = {}
        
    def set_event_loop(self, loop):
        """Set the asyncio event loop"""
        self.event_loop = loop
        
    def set_gui_callback(self, callback: Callable):
        """Set callback for GUI updates"""
        self.gui_callback = callback
        
    def set_trade_callback(self, callback: Callable):
        """Set callback for trade updates"""
        self.trade_callback = callback
        
    def set_status_callback(self, callback: Callable):
        """Set callback for status updates"""
        self.status_callback = callback
        
    async def connect(self) -> bool:
        """Connect to Interactive Brokers"""
        try:
            self.logger.info(f"Connecting to IBKR at {self.config.ibkr_host}:{self.config.ibkr_port}")
            
            await self.ib.connectAsync(
                host=self.config.ibkr_host,
                port=self.config.ibkr_port,
                clientId=self.config.client_id,
                timeout=self.config.timeout
            )
            
            self.is_connected = True
            self.logger.info("Successfully connected to IBKR")
            log_connection_event("CONNECTED", f"Client ID: {self.config.client_id}")
            
            # Get account information
            await self.update_account_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"IBKR connection failed: {e}")
            log_connection_event("CONNECTION_FAILED", str(e))
            self.is_connected = False
            return False
            
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        try:
            if self.is_connected:
                self.stop_streaming()
                self.ib.disconnect()
                self.is_connected = False
                self.logger.info("Disconnected from IBKR")
                log_connection_event("DISCONNECTED", "Clean disconnect")
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            
    async def update_account_info(self):
        """Update account information"""
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'TotalCashValue' and av.currency == 'USD':
                    self.account_balance = float(av.value)
                    break
                    
            self.logger.info(f"Account balance: ${self.account_balance:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Account info update error: {e}")
            
    async def start_streaming(self, tickers: List[str]):
        """Start real-time market data streaming"""
        try:
            if not self.is_connected:
                self.logger.error("Not connected to IBKR")
                return False
                
            self.logger.info(f"Starting streaming for {len(tickers)} tickers")
            
            # Clear existing data
            self.market_data.clear()
            self.historical_data.clear()
            
            # Create contracts and request data
            for ticker in tickers:
                if validate_ticker_symbol(ticker):
                    try:
                        # Create stock contract
                        contract = Stock(ticker, self.config.default_exchange, 'USD')
                        
                        # Qualify contract
                        qualified_contracts = await self.ib.qualifyContractsAsync(contract)
                        if qualified_contracts:
                            contract = qualified_contracts[0]
                            
                            # Request market data
                            self.ib.reqMktData(contract, '', False, False)
                            
                            # Get historical data
                            historical = await self.ib.reqHistoricalDataAsync(
                                contract,
                                endDateTime='',
                                durationStr=self.config.duration_str,
                                barSizeSetting=self.config.default_chart_interval,
                                whatToShow='TRADES',
                                useRTH=True
                            )
                            
                            if historical:
                                self.historical_data[ticker] = historical
                                
                                # Initialize market data
                                self.market_data[ticker] = {
                                    'contract': contract,
                                    'last': 0.0,
                                    'bid': 0.0,
                                    'ask': 0.0,
                                    'volume': 0,
                                    'timestamp': datetime.now(),
                                    'historical': historical
                                }
                                
                                self.logger.info(f"Started streaming for {ticker}")
                                
                    except Exception as e:
                        self.logger.error(f"Error setting up streaming for {ticker}: {e}")
                        continue
                        
            # Set up market data callbacks
            self.ib.tickerUpdateEvent += self.on_ticker_update
            
            self.is_streaming = True
            self.logger.info("Market data streaming started")
            
            # Start analysis loop
            asyncio.create_task(self.analysis_loop())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Streaming start error: {e}")
            return False
            
    def stop_streaming(self):
        """Stop market data streaming"""
        try:
            if self.is_streaming:
                # Cancel all market data subscriptions
                for ticker_data in self.market_data.values():
                    if 'contract' in ticker_data:
                        self.ib.cancelMktData(ticker_data['contract'])
                        
                self.is_streaming = False
                self.logger.info("Market data streaming stopped")
                
        except Exception as e:
            self.logger.error(f"Streaming stop error: {e}")
            
    def on_ticker_update(self, ticker):
        """Handle real-time ticker updates"""
        try:
            symbol = ticker.contract.symbol
            
            if symbol in self.market_data:
                # Update market data
                self.market_data[symbol].update({
                    'last': ticker.last if ticker.last and ticker.last > 0 else self.market_data[symbol].get('last', 0),
                    'bid': ticker.bid if ticker.bid and ticker.bid > 0 else self.market_data[symbol].get('bid', 0),
                    'ask': ticker.ask if ticker.ask and ticker.ask > 0 else self.market_data[symbol].get('ask', 0),
                    'volume': ticker.volume if ticker.volume else self.market_data[symbol].get('volume', 0),
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            self.logger.error(f"Ticker update error: {e}")
            
    async def analysis_loop(self):
        """Main analysis and trading loop"""
        while self.is_streaming:
            try:
                await self.analyze_and_trade()
                await asyncio.sleep(self.config.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(self.config.analysis_interval)
                
    async def analyze_and_trade(self):
        """Perform market analysis and execute trades"""
        try:
            analyzed_data = {}
            
            for symbol, data in self.market_data.items():
                if symbol not in self.historical_data:
                    continue
                    
                # Get historical data
                historical = self.historical_data[symbol]
                if len(historical) < self.config.bb_period:
                    continue
                    
                # Perform technical analysis
                analysis = self.technical_analyzer.analyze_symbol(symbol, historical, data)
                
                # Update adaptive thresholds
                self.update_adaptive_thresholds(symbol, analysis)
                
                # Check for trade signals
                trade_signal = self.evaluate_trade_signal(symbol, analysis)
                
                # Execute trade if signal is strong enough
                if trade_signal and self.should_execute_trade(symbol, trade_signal):
                    await self.execute_trade(symbol, trade_signal, analysis)
                    
                # Store analysis results
                analyzed_data[symbol] = analysis
                
            # Update GUI
            if self.gui_callback:
                self.gui_callback(analyzed_data)
                
        except Exception as e:
            self.logger.error(f"Analysis and trade error: {e}")
            
    def update_adaptive_thresholds(self, symbol: str, analysis: Dict):
        """Update adaptive thresholds based on market conditions"""
        try:
            if symbol not in self.adaptive_thresholds:
                self.adaptive_thresholds[symbol] = {
                    'rsi_oversold': self.config.rsi_oversold,
                    'rsi_overbought': self.config.rsi_overbought,
                    'bb_lower': self.config.bb_lower_threshold,
                    'bb_upper': self.config.bb_upper_threshold,
                    'stoch_oversold': self.config.stoch_oversold,
                    'stoch_overbought': self.config.stoch_overbought
                }
                
            # Get volatility measure
            volatility = analysis.get('volatility', {}).get('current', 0)
            
            # Adjust thresholds based on volatility
            volatility_factor = 1 + (volatility * self.config.volatility_adjustment)
            
            # Update RSI thresholds
            base_rsi_oversold = self.config.rsi_oversold
            base_rsi_overbought = self.config.rsi_overbought
            
            self.adaptive_thresholds[symbol]['rsi_oversold'] = max(20, base_rsi_oversold / volatility_factor)
            self.adaptive_thresholds[symbol]['rsi_overbought'] = min(80, base_rsi_overbought * volatility_factor)
            
            # Update Bollinger Band thresholds
            self.adaptive_thresholds[symbol]['bb_lower'] = max(0.05, self.config.bb_lower_threshold / volatility_factor)
            self.adaptive_thresholds[symbol]['bb_upper'] = min(0.95, self.config.bb_upper_threshold * volatility_factor)
            
        except Exception as e:
            self.logger.error(f"Adaptive threshold update error for {symbol}: {e}")
            
    def evaluate_trade_signal(self, symbol: str, analysis: Dict) -> Optional[Dict]:
        """Evaluate if conditions warrant a trade signal"""
        try:
            # Get adaptive thresholds
            thresholds = self.adaptive_thresholds.get(symbol, {})
            
            # Extract analysis results
            rsi = analysis.get('rsi', {}).get('value', 50)
            bb_position = analysis.get('bollinger', {}).get('position_percent', 0.5)
            stoch_k = analysis.get('stochastic', {}).get('k_percent', 50)
            stoch_d = analysis.get('stochastic', {}).get('d_percent', 50)
            slope = analysis.get('slope', {}).get('value', 0)
            price_change = analysis.get('price_change', {}).get('percent_change', 0)
            volume_ratio = analysis.get('volume_analysis', {}).get('volume_ratio', 1.0)
            
            # Check for spike conditions
            is_spike = abs(price_change) > self.config.spike_threshold and volume_ratio > self.config.spike_volume_multiplier
            
            # Buy signal conditions
            buy_conditions = [
                rsi < thresholds.get('rsi_oversold', self.config.rsi_oversold),
                bb_position < thresholds.get('bb_lower', self.config.bb_lower_threshold),
                stoch_k < thresholds.get('stoch_oversold', self.config.stoch_oversold),
                slope > self.config.slope_threshold,
                price_change < -self.config.spike_threshold,  # Negative spike (drop)
                volume_ratio > 1.5  # Above average volume
            ]
            
            # Sell signal conditions
            sell_conditions = [
                rsi > thresholds.get('rsi_overbought', self.config.rsi_overbought),
                bb_position > thresholds.get('bb_upper', self.config.bb_upper_threshold),
                stoch_k > thresholds.get('stoch_overbought', self.config.stoch_overbought),
                slope < -self.config.slope_threshold,
                price_change > self.config.spike_threshold,  # Positive spike (rise)
                volume_ratio > 1.5  # Above average volume
            ]
            
            # Calculate signal strength
            buy_strength = sum(buy_conditions) / len(buy_conditions)
            sell_strength = sum(sell_conditions) / len(sell_conditions)
            
            # Determine signal
            if buy_strength >= 0.6 or (is_spike and price_change < 0):  # 60% conditions met or negative spike
                return {
                    'action': 'BUY',
                    'strength': buy_strength,
                    'price': analysis.get('current_price', 0),
                    'is_spike': is_spike,
                    'conditions_met': buy_conditions
                }
            elif sell_strength >= 0.6 or (is_spike and price_change > 0):  # 60% conditions met or positive spike
                return {
                    'action': 'SELL',
                    'strength': sell_strength,
                    'price': analysis.get('current_price', 0),
                    'is_spike': is_spike,
                    'conditions_met': sell_conditions
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Trade signal evaluation error for {symbol}: {e}")
            return None
            
    def should_execute_trade(self, symbol: str, trade_signal: Dict) -> bool:
        """Determine if trade should be executed based on risk management"""
        try:
            # Check if we already have a position
            if symbol in self.open_positions:
                return False
                
            # Check daily limits
            if not self.risk_manager.check_daily_limits(self.daily_trades_count, self.daily_pnl):
                return False
                
            # Check cooldown
            if symbol in self.last_trade_time:
                time_since_last = datetime.now() - self.last_trade_time[symbol]
                if time_since_last < timedelta(minutes=self.config.cooldown_minutes):
                    return False
                    
            # Check force trade mode
            if self.force_trade_mode:
                force_cooldown_key = f"{symbol}_{trade_signal['action']}"
                if force_cooldown_key in self.force_trade_cooldown:
                    time_since_force = datetime.now() - self.force_trade_cooldown[force_cooldown_key]
                    if time_since_force < timedelta(minutes=self.config.force_trade_cooldown):
                        return False
                        
            # Check signal strength (lower threshold for force trade mode)
            min_strength = 0.4 if self.force_trade_mode else 0.6
            if trade_signal['strength'] < min_strength:
                return False
                
            # Check account balance
            if self.account_balance < self.config.equity_per_trade:
                log_risk_event("INSUFFICIENT_FUNDS", symbol, {
                    'account_balance': self.account_balance,
                    'required': self.config.equity_per_trade
                })
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Trade execution check error for {symbol}: {e}")
            return False
            
    async def execute_trade(self, symbol: str, trade_signal: Dict, analysis: Dict):
        """Execute a trade based on the signal"""
        try:
            action = trade_signal['action']
            price = trade_signal['price']
            
            # Calculate position size
            position_size = calculate_position_size(
                self.config.equity_per_trade,
                price,
                self.config.hard_stop_loss
            )
            
            if position_size <= 0:
                return
                
            # Create contract
            contract = self.market_data[symbol]['contract']
            
            # Create order
            if action == 'BUY':
                order = MarketOrder('BUY', position_size)
            else:  # SELL (short)
                order = MarketOrder('SELL', position_size)
                
            # Calculate stop loss and take profit
            if action == 'BUY':
                stop_loss_price = price * (1 - self.config.hard_stop_loss)
                take_profit_price = price * (1 + self.config.take_profit)
            else:
                stop_loss_price = price * (1 + self.config.hard_stop_loss)
                take_profit_price = price * (1 - self.config.take_profit)
                
            # Execute main order
            trade = self.ib.placeOrder(contract, order)
            
            # Generate trade ID
            trade_id = generate_trade_id()
            
            # Log trade
            log_trade_event(f"{action}_ORDER", symbol, {
                'trade_id': trade_id,
                'quantity': position_size,
                'price': price,
                'signal_strength': trade_signal['strength'],
                'is_spike': trade_signal.get('is_spike', False),
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            })
            
            # Store position
            self.open_positions[symbol] = {
                'trade_id': trade_id,
                'action': action,
                'quantity': position_size,
                'entry_price': price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'entry_time': datetime.now(),
                'trade_object': trade,
                'analysis': analysis
            }
            
            # Update counters
            self.daily_trades_count += 1
            self.last_trade_time[symbol] = datetime.now()
            
            # Update force trade cooldown
            if self.force_trade_mode:
                force_cooldown_key = f"{symbol}_{action}"
                self.force_trade_cooldown[force_cooldown_key] = datetime.now()
                
            # Create stop loss and take profit orders
            await self.create_exit_orders(symbol, contract, stop_loss_price, take_profit_price)
            
            # Update GUI
            if self.trade_callback:
                self.trade_callback(self.open_positions, self.trade_history)
                
            self.logger.info(f"Executed {action} trade for {symbol}: {position_size} @ ${price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Trade execution error for {symbol}: {e}")
            log_trade_event("EXECUTION_ERROR", symbol, {'error': str(e)})
            
    async def create_exit_orders(self, symbol: str, contract, stop_loss_price: float, take_profit_price: float):
        """Create stop loss and take profit orders"""
        try:
            position = self.open_positions[symbol]
            action = position['action']
            quantity = position['quantity']
            
            # Create stop loss order
            if action == 'BUY':
                stop_order = StopOrder('SELL', quantity, stop_loss_price)
                limit_order = LimitOrder('SELL', quantity, take_profit_price)
            else:
                stop_order = StopOrder('BUY', quantity, stop_loss_price)
                limit_order = LimitOrder('BUY', quantity, take_profit_price)
                
            # Place orders
            stop_trade = self.ib.placeOrder(contract, stop_order)
            limit_trade = self.ib.placeOrder(contract, limit_order)
            
            # Store order references
            position['stop_order'] = stop_trade
            position['limit_order'] = limit_trade
            
        except Exception as e:
            self.logger.error(f"Exit order creation error for {symbol}: {e}")
            
    def toggle_force_trade_mode(self):
        """Toggle force trade mode"""
        self.force_trade_mode = not self.force_trade_mode
        self.logger.info(f"Force trade mode: {'ON' if self.force_trade_mode else 'OFF'}")
        
    def check_force_trade_conditions(self):
        """Check and update force trade conditions"""
        try:
            # Clean up expired cooldowns
            current_time = datetime.now()
            expired_cooldowns = []
            
            for key, cooldown_time in self.force_trade_cooldown.items():
                if current_time - cooldown_time > timedelta(minutes=self.config.force_trade_cooldown):
                    expired_cooldowns.append(key)
                    
            for key in expired_cooldowns:
                del self.force_trade_cooldown[key]
                
        except Exception as e:
            self.logger.error(f"Force trade condition check error: {e}")
            
    def update_cooldown_timers(self):
        """Update and clean up cooldown timers"""
        try:
            current_time = datetime.now()
            expired_timers = []
            
            for symbol, last_time in self.last_trade_time.items():
                if current_time - last_time > timedelta(minutes=self.config.cooldown_minutes):
                    expired_timers.append(symbol)
                    
            for symbol in expired_timers:
                del self.last_trade_time[symbol]
                
        except Exception as e:
            self.logger.error(f"Cooldown timer update error: {e}")
            
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if not self.trade_history:
                return
                
            # Calculate basic metrics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            losing_trades = total_trades - winning_trades
            
            total_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate daily P&L
            today = datetime.now().date()
            daily_pnl = sum(
                trade.get('pnl', 0) for trade in self.trade_history
                if trade.get('exit_time', datetime.now()).date() == today
            )
            
            # Update metrics
            self.performance_metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate
            })
            
            self.daily_pnl = daily_pnl
            
        except Exception as e:
            self.logger.error(f"Performance metrics update error: {e}")
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.performance_metrics.get('total_pnl', 0),
            'win_rate': self.performance_metrics.get('win_rate', 0),
            'total_trades': self.performance_metrics.get('total_trades', 0),
            'daily_trades': self.daily_trades_count,
            'open_positions': len(self.open_positions),
            'account_balance': self.account_balance,
            'force_trade_mode': self.force_trade_mode
        }
