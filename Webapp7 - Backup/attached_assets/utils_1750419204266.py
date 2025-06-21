"""
Enhanced utility functions for the sophisticated trading application
Integrated from reserve_file.py with complete calculation methods
"""

import logging
from datetime import datetime, timedelta
from typing import Union, Optional, List, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
import re

def calculate_position_size(equity_per_trade: float, price: float, stop_loss_percent: float) -> int:
    """
    Calculate position size based on equity allocation and stop loss
    
    Args:
        equity_per_trade: Dollar amount to risk per trade
        price: Current stock price
        stop_loss_percent: Stop loss as a decimal (e.g., 0.03 for 3%)
    
    Returns:
        Number of shares to buy
    """
    try:
        if price <= 0 or stop_loss_percent <= 0:
            return 0
            
        # Calculate risk per share
        risk_per_share = price * stop_loss_percent
        
        # Calculate maximum shares based on risk tolerance
        max_shares = int(equity_per_trade / risk_per_share)
        
        return max_shares
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Position size calculation error: {e}")
        return 0

def format_currency(amount: float, symbol: str = "$") -> str:
    """Format currency with proper symbol and comma separators"""
    try:
        if amount >= 0:
            return f"{symbol}{amount:,.2f}"
        else:
            return f"-{symbol}{abs(amount):,.2f}"
    except:
        return f"{symbol}0.00"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage with proper sign and decimals"""
    try:
        if value >= 0:
            return f"+{value:.{decimals}f}%"
        else:
            return f"{value:.{decimals}f}%"
    except:
        return "0.00%"

def calculate_time_difference(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """Calculate human-readable time difference"""
    try:
        if end_time is None:
            end_time = datetime.now()
            
        diff = end_time - start_time
        
        if diff.days > 0:
            return f"{diff.days}d {diff.seconds // 3600}h"
        elif diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h {(diff.seconds % 3600) // 60}m"
        elif diff.seconds >= 60:
            return f"{diff.seconds // 60}m {diff.seconds % 60}s"
        else:
            return f"{diff.seconds}s"
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Time difference calculation error: {e}")
        return "N/A"

def validate_ticker_symbol(symbol: str) -> bool:
    """Validate ticker symbol format"""
    try:
        if not symbol or not isinstance(symbol, str):
            return False
            
        # Remove whitespace and convert to uppercase
        symbol = symbol.strip().upper()
        
        # Basic validation: letters only, 1-5 characters
        if not symbol.isalpha() or not (1 <= len(symbol) <= 5):
            return False
            
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Ticker validation error: {e}")
        return False

def parse_ticker_list(tickers_string: str) -> List[str]:
    """Parse comma-separated ticker string into validated list"""
    try:
        if not tickers_string:
            return []
            
        # Split by comma and clean up
        tickers = [t.strip().upper() for t in tickers_string.split(',')]
        
        # Validate each ticker
        valid_tickers = [t for t in tickers if validate_ticker_symbol(t)]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []
        for ticker in valid_tickers:
            if ticker not in seen:
                seen.add(ticker)
                unique_tickers.append(ticker)
                
        return unique_tickers
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Ticker list parsing error: {e}")
        return []

def convert_timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds"""
    try:
        timeframe = timeframe.strip().lower()
        
        mapping = {
            '1 sec': 1,
            '5 sec': 5,
            '10 sec': 10,
            '30 sec': 30,
            '1 min': 60,
            '3 mins': 180,
            '5 mins': 300,
            '10 mins': 600,
            '15 mins': 900,
            '30 mins': 1800,
            '1 hour': 3600,
            '2 hours': 7200,
            '4 hours': 14400,
            '1 day': 86400
        }
        
        return mapping.get(timeframe, 300)  # Default to 5 minutes
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Timeframe conversion error: {e}")
        return 300

def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, take_profit: float) -> float:
    """Calculate risk-to-reward ratio"""
    try:
        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return 0
            
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0
            
        return reward / risk
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Risk-reward calculation error: {e}")
        return 0

def calculate_compound_return(initial_balance: float, trades: List[Dict]) -> Dict[str, float]:
    """Calculate compound return from a series of trades"""
    try:
        if not trades or initial_balance <= 0:
            return {'final_balance': initial_balance, 'total_return': 0, 'compound_rate': 0}
            
        current_balance = initial_balance
        
        for trade in trades:
            pnl = trade.get('final_pnl', 0)
            current_balance += pnl
            
        total_return = current_balance - initial_balance
        compound_rate = (current_balance / initial_balance - 1) * 100
        
        return {
            'initial_balance': initial_balance,
            'final_balance': current_balance,
            'total_return': total_return,
            'compound_rate': compound_rate,
            'number_of_trades': len(trades)
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Compound return calculation error: {e}")
        return {'final_balance': initial_balance, 'total_return': 0, 'compound_rate': 0}

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns list"""
    try:
        if not returns or len(returns) < 2:
            return 0
            
        returns_array = np.array(returns)
        
        # Convert to pandas Series for easier calculation
        returns_series = pd.Series(returns_array)
        
        # Calculate excess returns
        excess_returns = returns_series - (risk_free_rate / 252)  # Daily risk-free rate
        
        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            return 0
            
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
        
        return sharpe_ratio
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Sharpe ratio calculation error: {e}")
        return 0

def calculate_max_drawdown(equity_curve: List[float]) -> Dict[str, float]:
    """Calculate maximum drawdown from equity curve"""
    try:
        if not equity_curve or len(equity_curve) < 2:
            return {'max_drawdown': 0, 'max_drawdown_percent': 0}
            
        equity_array = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Calculate drawdown
        drawdown = equity_array - running_max
        
        # Find maximum drawdown
        max_drawdown = np.min(drawdown)
        max_drawdown_percent = (max_drawdown / np.max(running_max)) * 100 if np.max(running_max) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_percent,
            'drawdown_series': drawdown.tolist()
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Max drawdown calculation error: {e}")
        return {'max_drawdown': 0, 'max_drawdown_percent': 0}

def is_market_open(current_time: Optional[datetime] = None) -> bool:
    """Check if market is currently open (basic implementation)"""
    try:
        if current_time is None:
            current_time = datetime.now()
            
        # Check if weekend
        if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Market hours: 9:30 AM - 4:00 PM ET (simplified)
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= current_time <= market_close
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Market hours check error: {e}")
        return True  # Default to market open for safety

def generate_trade_id() -> str:
    """Generate unique trade identifier"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"TRADE_{timestamp}"
    except Exception as e:
        logging.getLogger(__name__).error(f"Trade ID generation error: {e}")
        return f"TRADE_{int(datetime.now().timestamp())}"

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value between min and max"""
    try:
        return max(min_value, min(value, max_value))
    except:
        return min_value

def exponential_moving_average(data: List[float], period: int) -> List[float]:
    """Calculate exponential moving average"""
    try:
        if not data or period <= 0:
            return []
            
        # Convert to pandas Series for easier calculation
        series = pd.Series(data)
        ema = series.ewm(span=period).mean()
        
        return ema.tolist()
        
    except Exception as e:
        logging.getLogger(__name__).error(f"EMA calculation error: {e}")
        return data  # Return original data if calculation fails

# Enhanced Technical Analysis Functions

def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    Compute RSI (Relative Strength Index) - Enhanced version from reserve_file.py
    
    Args:
        prices: Array of closing prices
        period: RSI period (default 14)
    
    Returns:
        Current RSI value
    """
    try:
        if len(prices) < period + 1:
            return 50.0
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate subsequent values using Wilder's smoothing
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
        # Calculate RSI
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logging.getLogger(__name__).error(f"RSI computation error: {e}")
        return 50.0

def compute_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """
    Compute Bollinger Bands - Enhanced version from reserve_file.py
    
    Args:
        prices: Array of closing prices
        period: Moving average period
        std_dev: Standard deviation multiplier
    
    Returns:
        Dictionary with upper, middle, lower bands and current position
    """
    try:
        if len(prices) < period:
            current_price = prices[-1] if len(prices) > 0 else 0
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98,
                'position_percent': 0.5
            }
            
        # Calculate moving average (middle band)
        middle_band = np.mean(prices[-period:])
        
        # Calculate standard deviation
        std = np.std(prices[-period:])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Calculate current position within bands
        current_price = prices[-1]
        if upper_band > lower_band:
            position_percent = (current_price - lower_band) / (upper_band - lower_band)
        else:
            position_percent = 0.5
            
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'position_percent': position_percent,
            'band_width': upper_band - lower_band,
            'current_price': current_price
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Bollinger Bands computation error: {e}")
        current_price = prices[-1] if len(prices) > 0 else 0
        return {
            'upper': current_price * 1.02,
            'middle': current_price,
            'lower': current_price * 0.98,
            'position_percent': 0.5
        }

def compute_stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                      k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
    """
    Compute Stochastic Oscillator - Enhanced version from reserve_file.py
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        k_period: %K period
        d_period: %D period
    
    Returns:
        Dictionary with %K and %D values
    """
    try:
        if len(closes) < k_period:
            return {
                'k_percent': 50.0,
                'd_percent': 50.0,
                'oversold': False,
                'overbought': False
            }
            
        # Calculate %K values
        k_values = []
        for i in range(k_period - 1, len(closes)):
            period_high = np.max(highs[i - k_period + 1:i + 1])
            period_low = np.min(lows[i - k_period + 1:i + 1])
            
            if period_high > period_low:
                k_percent = ((closes[i] - period_low) / (period_high - period_low)) * 100
            else:
                k_percent = 50.0
                
            k_values.append(k_percent)
            
        if not k_values:
            return {
                'k_percent': 50.0,
                'd_percent': 50.0,
                'oversold': False,
                'overbought': False
            }
            
        current_k = k_values[-1]
        
        # Calculate %D (SMA of %K)
        if len(k_values) >= d_period:
            current_d = np.mean(k_values[-d_period:])
        else:
            current_d = current_k
            
        return {
            'k_percent': current_k,
            'd_percent': current_d,
            'oversold': current_k < 20,
            'overbought': current_k > 80,
            'bullish_crossover': current_k > current_d,
            'bearish_crossover': current_k < current_d
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Stochastic computation error: {e}")
        return {
            'k_percent': 50.0,
            'd_percent': 50.0,
            'oversold': False,
            'overbought': False
        }

def compute_slope(prices: np.ndarray, period: int = 10) -> Dict[str, float]:
    """
    Compute price slope (trend direction and strength) - Enhanced version from reserve_file.py
    
    Args:
        prices: Array of closing prices
        period: Period for slope calculation
    
    Returns:
        Dictionary with slope value and trend information
    """
    try:
        if len(prices) < period:
            return {
                'value': 0.0,
                'trend': 'neutral',
                'strength': 'weak',
                'r_squared': 0.0
            }
            
        # Get recent prices
        recent_prices = prices[-period:]
        x = np.arange(len(recent_prices))
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
        
        # Normalize slope by average price
        avg_price = np.mean(recent_prices)
        normalized_slope = (slope / avg_price) * 100 if avg_price > 0 else 0
        
        # Determine trend
        if normalized_slope > 0.02:  # 2% threshold
            trend = 'bullish'
        elif normalized_slope < -0.02:
            trend = 'bearish'
        else:
            trend = 'neutral'
            
        # Determine strength based on R-squared
        r_squared = r_value ** 2
        if r_squared > 0.7:
            strength = 'strong'
        elif r_squared > 0.4:
            strength = 'medium'
        else:
            strength = 'weak'
        
        return {
            'value': normalized_slope,
            'trend': trend,
            'strength': strength,
            'r_squared': r_squared,
            'slope': slope,
            'intercept': intercept
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Slope computation error: {e}")
        return {
            'value': 0.0,
            'trend': 'neutral',
            'strength': 'weak',
            'r_squared': 0.0
        }

def compute_volume_analysis(volumes: np.ndarray, current_volume: int, period: int = 20) -> Dict[str, Any]:
    """
    Compute volume analysis
    
    Args:
        volumes: Array of historical volumes
        current_volume: Current period volume
        period: Period for average calculation
    
    Returns:
        Dictionary with volume analysis
    """
    try:
        if len(volumes) < period:
            return {
                'current_volume': current_volume,
                'avg_volume': current_volume,
                'volume_ratio': 1.0,
                'unusual_volume': False,
                'volume_trend': 'normal'
            }
            
        # Calculate average volume
        avg_volume = np.mean(volumes[-period:])
        
        # Calculate volume ratio
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Detect unusual volume
        unusual_volume = volume_ratio > 2.0  # 2x average
        
        # Determine volume trend
        if volume_ratio > 1.5:
            volume_trend = 'increasing'
        elif volume_ratio < 0.7:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'normal'
            
        return {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'unusual_volume': unusual_volume,
            'volume_trend': volume_trend,
            'volume_percentile': stats.percentileofscore(volumes[-period:], current_volume)
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Volume analysis error: {e}")
        return {
            'current_volume': current_volume,
            'avg_volume': current_volume,
            'volume_ratio': 1.0,
            'unusual_volume': False,
            'volume_trend': 'normal'
        }

def compute_adaptive_thresholds(data: np.ndarray, lookback: int = 50, percentile: float = 0.8) -> Dict[str, float]:
    """
    Compute adaptive thresholds based on historical data
    
    Args:
        data: Historical data array
        lookback: Lookback period
        percentile: Percentile for threshold calculation
    
    Returns:
        Dictionary with adaptive thresholds
    """
    try:
        if len(data) < lookback:
            return {
                'upper_threshold': np.percentile(data, 80) if len(data) > 0 else 0,
                'lower_threshold': np.percentile(data, 20) if len(data) > 0 else 0,
                'volatility_factor': 1.0
            }
            
        recent_data = data[-lookback:]
        
        # Calculate percentile-based thresholds
        upper_threshold = np.percentile(recent_data, percentile * 100)
        lower_threshold = np.percentile(recent_data, (1 - percentile) * 100)
        
        # Calculate volatility factor
        volatility = np.std(recent_data)
        avg_volatility = np.std(data) if len(data) > lookback else volatility
        volatility_factor = volatility / avg_volatility if avg_volatility > 0 else 1.0
        
        return {
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'volatility_factor': volatility_factor,
            'current_volatility': volatility,
            'avg_volatility': avg_volatility
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Adaptive thresholds computation error: {e}")
        return {
            'upper_threshold': 0,
            'lower_threshold': 0,
            'volatility_factor': 1.0
        }

def detect_price_spike(prices: np.ndarray, volumes: np.ndarray, 
                      price_threshold: float = 0.02, volume_multiplier: float = 2.0) -> Dict[str, Any]:
    """
    Detect price spikes with volume confirmation
    
    Args:
        prices: Array of prices
        volumes: Array of volumes
        price_threshold: Price change threshold for spike detection
        volume_multiplier: Volume multiplier for confirmation
    
    Returns:
        Dictionary with spike detection results
    """
    try:
        if len(prices) < 2 or len(volumes) < 20:
            return {
                'is_spike': False,
                'spike_direction': 'none',
                'price_change': 0.0,
                'volume_ratio': 1.0,
                'confirmed_spike': False
            }
            
        # Calculate price change
        price_change = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
        
        # Calculate volume ratio
        avg_volume = np.mean(volumes[-20:-1])  # Exclude current period
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Detect spike
        is_price_spike = abs(price_change) > price_threshold
        is_volume_spike = volume_ratio > volume_multiplier
        
        # Determine spike direction
        if price_change > price_threshold:
            spike_direction = 'up'
        elif price_change < -price_threshold:
            spike_direction = 'down'
        else:
            spike_direction = 'none'
            
        # Confirm spike with volume
        confirmed_spike = is_price_spike and is_volume_spike
        
        return {
            'is_spike': is_price_spike,
            'spike_direction': spike_direction,
            'price_change': price_change,
            'volume_ratio': volume_ratio,
            'confirmed_spike': confirmed_spike,
            'price_threshold': price_threshold,
            'volume_threshold': volume_multiplier
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Price spike detection error: {e}")
        return {
            'is_spike': False,
            'spike_direction': 'none',
            'price_change': 0.0,
            'volume_ratio': 1.0,
            'confirmed_spike': False
        }

def calculate_trade_score(rsi: float, bb_position: float, stoch_k: float, 
                         slope: float, volume_ratio: float, 
                         adaptive_thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate comprehensive trade score based on multiple indicators
    
    Args:
        rsi: RSI value
        bb_position: Bollinger Band position (0-1)
        stoch_k: Stochastic %K
        slope: Price slope
        volume_ratio: Volume ratio
        adaptive_thresholds: Adaptive threshold values
    
    Returns:
        Dictionary with trade scores and signals
    """
    try:
        # Buy signal conditions
        buy_conditions = {
            'rsi_oversold': rsi < adaptive_thresholds.get('rsi_oversold', 30),
            'bb_lower': bb_position < adaptive_thresholds.get('bb_lower', 0.2),
            'stoch_oversold': stoch_k < adaptive_thresholds.get('stoch_oversold', 20),
            'positive_slope': slope > 0.01,
            'volume_confirmation': volume_ratio > 1.2
        }
        
        # Sell signal conditions
        sell_conditions = {
            'rsi_overbought': rsi > adaptive_thresholds.get('rsi_overbought', 70),
            'bb_upper': bb_position > adaptive_thresholds.get('bb_upper', 0.8),
            'stoch_overbought': stoch_k > adaptive_thresholds.get('stoch_overbought', 80),
            'negative_slope': slope < -0.01,
            'volume_confirmation': volume_ratio > 1.2
        }
        
        # Calculate scores
        buy_score = sum(buy_conditions.values()) / len(buy_conditions)
        sell_score = sum(sell_conditions.values()) / len(sell_conditions)
        
        # Determine overall signal
        if buy_score >= 0.6:
            overall_signal = 'BUY'
            confidence = buy_score
        elif sell_score >= 0.6:
            overall_signal = 'SELL'
            confidence = sell_score
        else:
            overall_signal = 'HOLD'
            confidence = max(buy_score, sell_score)
            
        return {
            'buy_score': buy_score,
            'sell_score': sell_score,
            'overall_signal': overall_signal,
            'confidence': confidence,
            'buy_conditions': buy_conditions,
            'sell_conditions': sell_conditions,
            'conditions_met': sum(buy_conditions.values()) + sum(sell_conditions.values())
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Trade score calculation error: {e}")
        return {
            'buy_score': 0.0,
            'sell_score': 0.0,
            'overall_signal': 'HOLD',
            'confidence': 0.0,
            'buy_conditions': {},
            'sell_conditions': {},
            'conditions_met': 0
        }
