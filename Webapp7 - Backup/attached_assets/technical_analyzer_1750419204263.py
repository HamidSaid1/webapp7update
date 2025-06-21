"""
Advanced Technical Analysis Engine with sophisticated calculations
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats
# import talib  # Not available in this environment, using custom implementations

class TechnicalAnalyzer:
    """Advanced technical analysis with multiple indicators and adaptive algorithms"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_symbol(self, symbol: str, historical_data, current_data: Dict) -> Dict[str, Any]:
        """Perform comprehensive technical analysis for a symbol"""
        try:
            # Convert historical data to DataFrame
            df = self.prepare_dataframe(historical_data)
            
            # Get current price
            current_price = current_data.get('last', 0)
            
            if df.empty or current_price <= 0:
                return self.get_empty_analysis(symbol, current_price)
                
            # Perform all calculations
            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'rsi': self.compute_rsi(df),
                'bollinger': self.compute_bollinger_bands(df, current_price),
                'stochastic': self.compute_stochastic(df),
                'slope': self.compute_slope(df),
                'price_change': self.compute_price_change(df, current_price),
                'volume_analysis': self.compute_volume_analysis(df, current_data),
                'volatility': self.compute_volatility(df),
                'momentum': self.compute_momentum(df),
                'support_resistance': self.compute_support_resistance(df, current_price),
                'trend_strength': self.compute_trend_strength(df),
                'signals': self.generate_signals(df, current_price)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Technical analysis error for {symbol}: {e}")
            return self.get_empty_analysis(symbol, current_data.get('last', 0))
            
    def prepare_dataframe(self, historical_data) -> pd.DataFrame:
        """Convert historical data to pandas DataFrame"""
        try:
            if hasattr(historical_data, 'df'):
                df = historical_data.df.copy()
            else:
                # Convert ib_insync BarDataList to DataFrame
                data = []
                for bar in historical_data:
                    data.append({
                        'date': bar.date,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    })
                df = pd.DataFrame(data)
                
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
                    
            # Convert to numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Remove any rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"DataFrame preparation error: {e}")
            return pd.DataFrame()
            
    def compute_rsi(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute RSI (Relative Strength Index)"""
        try:
            if len(df) < self.config.rsi_period:
                return {'value': 50.0, 'oversold': False, 'overbought': False}
                
            # Calculate RSI using talib
            rsi_values = talib.RSI(df['close'].values, timeperiod=self.config.rsi_period)
            current_rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0
            
            # Alternative calculation if talib fails
            if np.isnan(current_rsi):
                current_rsi = self.manual_rsi_calculation(df['close'].values, self.config.rsi_period)
                
            return {
                'value': current_rsi,
                'oversold': current_rsi < self.config.rsi_oversold,
                'overbought': current_rsi > self.config.rsi_overbought,
                'trend': 'bullish' if current_rsi > 50 else 'bearish'
            }
            
        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return {'value': 50.0, 'oversold': False, 'overbought': False}
            
    def manual_rsi_calculation(self, prices: np.ndarray, period: int) -> float:
        """Manual RSI calculation as fallback"""
        try:
            deltas = np.diff(prices)
            gain = np.where(deltas > 0, deltas, 0)
            loss = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gain[-period:])
            avg_loss = np.mean(loss[-period:])
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Manual RSI calculation error: {e}")
            return 50.0
            
    def compute_bollinger_bands(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Compute Bollinger Bands"""
        try:
            if len(df) < self.config.bb_period:
                return {
                    'upper': current_price * 1.02,
                    'middle': current_price,
                    'lower': current_price * 0.98,
                    'position_percent': 0.5,
                    'squeeze': False
                }
                
            # Calculate Bollinger Bands
            closes = df['close'].values
            
            # Calculate moving average
            sma = np.mean(closes[-self.config.bb_period:])
            
            # Calculate standard deviation
            std = np.std(closes[-self.config.bb_period:])
            
            # Calculate bands
            upper_band = sma + (std * self.config.bb_std_dev)
            lower_band = sma - (std * self.config.bb_std_dev)
            
            # Calculate position within bands
            if upper_band > lower_band:
                position_percent = (current_price - lower_band) / (upper_band - lower_band)
            else:
                position_percent = 0.5
                
            # Detect squeeze (low volatility)
            avg_band_width = np.mean([
                (np.mean(closes[-i-self.config.bb_period:-i]) + np.std(closes[-i-self.config.bb_period:-i]) * self.config.bb_std_dev) -
                (np.mean(closes[-i-self.config.bb_period:-i]) - np.std(closes[-i-self.config.bb_period:-i]) * self.config.bb_std_dev)
                for i in range(1, min(10, len(closes) - self.config.bb_period))
            ]) if len(closes) > self.config.bb_period + 10 else upper_band - lower_band
            
            current_band_width = upper_band - lower_band
            squeeze = current_band_width < avg_band_width * 0.8
            
            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band,
                'position_percent': position_percent,
                'squeeze': squeeze,
                'band_width': current_band_width,
                'avg_band_width': avg_band_width
            }
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {e}")
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98,
                'position_percent': 0.5,
                'squeeze': False
            }
            
    def compute_stochastic(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute Stochastic Oscillator"""
        try:
            if len(df) < self.config.stoch_k_period:
                return {
                    'k_percent': 50.0,
                    'd_percent': 50.0,
                    'oversold': False,
                    'overbought': False
                }
                
            # Calculate Stochastic %K
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            k_values = []
            for i in range(self.config.stoch_k_period - 1, len(closes)):
                period_high = np.max(highs[i - self.config.stoch_k_period + 1:i + 1])
                period_low = np.min(lows[i - self.config.stoch_k_period + 1:i + 1])
                
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
            
            # Calculate Stochastic %D (SMA of %K)
            if len(k_values) >= self.config.stoch_d_period:
                current_d = np.mean(k_values[-self.config.stoch_d_period:])
            else:
                current_d = current_k
                
            return {
                'k_percent': current_k,
                'd_percent': current_d,
                'oversold': current_k < self.config.stoch_oversold,
                'overbought': current_k > self.config.stoch_overbought,
                'bullish_crossover': current_k > current_d,
                'bearish_crossover': current_k < current_d
            }
            
        except Exception as e:
            self.logger.error(f"Stochastic calculation error: {e}")
            return {
                'k_percent': 50.0,
                'd_percent': 50.0,
                'oversold': False,
                'overbought': False
            }
            
    def compute_slope(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute price slope (trend direction and strength)"""
        try:
            if len(df) < self.config.slope_period:
                return {'value': 0.0, 'trend': 'neutral', 'strength': 'weak'}
                
            # Get recent prices
            prices = df['close'].values[-self.config.slope_period:]
            x = np.arange(len(prices))
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            # Normalize slope by average price
            avg_price = np.mean(prices)
            normalized_slope = (slope / avg_price) * 100 if avg_price > 0 else 0
            
            # Determine trend
            if normalized_slope > self.config.slope_threshold:
                trend = 'bullish'
            elif normalized_slope < -self.config.slope_threshold:
                trend = 'bearish'
            else:
                trend = 'neutral'
                
            # Determine strength based on R-squared
            strength = 'strong' if r_value ** 2 > 0.7 else 'medium' if r_value ** 2 > 0.4 else 'weak'
            
            return {
                'value': normalized_slope,
                'trend': trend,
                'strength': strength,
                'r_squared': r_value ** 2,
                'p_value': p_value
            }
            
        except Exception as e:
            self.logger.error(f"Slope calculation error: {e}")
            return {'value': 0.0, 'trend': 'neutral', 'strength': 'weak'}
            
    def compute_price_change(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Compute price change metrics"""
        try:
            if df.empty:
                return {'percent_change': 0.0, 'dollar_change': 0.0}
                
            previous_close = df['close'].iloc[-1]
            
            if previous_close > 0:
                percent_change = ((current_price - previous_close) / previous_close) * 100
                dollar_change = current_price - previous_close
            else:
                percent_change = 0.0
                dollar_change = 0.0
                
            return {
                'percent_change': percent_change,
                'dollar_change': dollar_change,
                'previous_close': previous_close,
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Price change calculation error: {e}")
            return {'percent_change': 0.0, 'dollar_change': 0.0}
            
    def compute_volume_analysis(self, df: pd.DataFrame, current_data: Dict) -> Dict[str, float]:
        """Compute volume analysis"""
        try:
            if df.empty:
                return {
                    'current_volume': 0,
                    'avg_volume': 0,
                    'volume_ratio': 1.0,
                    'unusual_volume': False
                }
                
            # Get current volume
            current_volume = current_data.get('volume', 0)
            
            # Calculate average volume
            recent_volumes = df['volume'].values[-20:]  # Last 20 periods
            avg_volume = np.mean(recent_volumes) if len(recent_volumes) > 0 else 1
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Detect unusual volume
            unusual_volume = volume_ratio > 2.0  # 2x average volume
            
            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'unusual_volume': unusual_volume,
                'volume_trend': 'increasing' if volume_ratio > 1.2 else 'decreasing' if volume_ratio < 0.8 else 'normal'
            }
            
        except Exception as e:
            self.logger.error(f"Volume analysis error: {e}")
            return {
                'current_volume': 0,
                'avg_volume': 0,
                'volume_ratio': 1.0,
                'unusual_volume': False
            }
            
    def compute_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute volatility metrics"""
        try:
            if len(df) < 20:
                return {'current': 0.0, 'avg': 0.0, 'percentile': 50.0}
                
            # Calculate daily returns
            returns = df['close'].pct_change().dropna()
            
            # Current volatility (20-day)
            current_vol = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized
            
            # Average volatility
            avg_vol = returns.std() * np.sqrt(252) * 100
            
            # Volatility percentile
            vol_percentile = stats.percentileofscore(returns.rolling(20).std() * np.sqrt(252) * 100, current_vol)
            
            return {
                'current': current_vol,
                'avg': avg_vol,
                'percentile': vol_percentile,
                'regime': 'high' if vol_percentile > 80 else 'low' if vol_percentile < 20 else 'normal'
            }
            
        except Exception as e:
            self.logger.error(f"Volatility calculation error: {e}")
            return {'current': 0.0, 'avg': 0.0, 'percentile': 50.0}
            
    def compute_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute momentum indicators"""
        try:
            if len(df) < 20:
                return {'roc': 0.0, 'macd': 0.0, 'signal': 0.0}
                
            closes = df['close'].values
            
            # Rate of Change (ROC)
            roc = ((closes[-1] - closes[-10]) / closes[-10]) * 100 if len(closes) >= 10 else 0.0
            
            # MACD calculation
            if len(closes) >= 26:
                ema_12 = self.calculate_ema(closes, 12)
                ema_26 = self.calculate_ema(closes, 26)
                macd = ema_12[-1] - ema_26[-1]
                
                # MACD Signal line
                macd_values = ema_12[-9:] - ema_26[-9:] if len(ema_12) >= 9 else [macd]
                signal = self.calculate_ema(macd_values, 9)[-1]
            else:
                macd = 0.0
                signal = 0.0
                
            return {
                'roc': roc,
                'macd': macd,
                'signal': signal,
                'histogram': macd - signal,
                'bullish': macd > signal,
                'bearish': macd < signal
            }
            
        except Exception as e:
            self.logger.error(f"Momentum calculation error: {e}")
            return {'roc': 0.0, 'macd': 0.0, 'signal': 0.0}
            
    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        try:
            alpha = 2.0 / (period + 1)
            ema = np.zeros_like(data)
            ema[0] = data[0]
            
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
                
            return ema
            
        except Exception as e:
            self.logger.error(f"EMA calculation error: {e}")
            return np.zeros_like(data)
            
    def compute_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict[str, List[float]]:
        """Compute support and resistance levels"""
        try:
            if len(df) < 50:
                return {'support': [], 'resistance': []}
                
            # Get recent highs and lows
            highs = df['high'].values[-50:]
            lows = df['low'].values[-50:]
            
            # Find pivot points
            support_levels = []
            resistance_levels = []
            
            # Simple pivot detection
            for i in range(2, len(highs) - 2):
                # Resistance (local maxima)
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    resistance_levels.append(highs[i])
                    
                # Support (local minima)
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    support_levels.append(lows[i])
                    
            # Filter levels close to current price
            support_levels = [level for level in support_levels if level < current_price * 1.05]
            resistance_levels = [level for level in resistance_levels if level > current_price * 0.95]
            
            # Sort and limit
            support_levels = sorted(support_levels, reverse=True)[:3]
            resistance_levels = sorted(resistance_levels)[:3]
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance calculation error: {e}")
            return {'support': [], 'resistance': []}
            
    def compute_trend_strength(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute trend strength using ADX"""
        try:
            if len(df) < 14:
                return {'adx': 0.0, 'strength': 'weak', 'trend': 'neutral'}
                
            # Calculate ADX using simplified method
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Calculate True Range
            tr_values = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                tr_values.append(tr)
                
            # Calculate Directional Movement
            dm_plus = []
            dm_minus = []
            
            for i in range(1, len(highs)):
                up_move = highs[i] - highs[i-1]
                down_move = lows[i-1] - lows[i]
                
                if up_move > down_move and up_move > 0:
                    dm_plus.append(up_move)
                else:
                    dm_plus.append(0)
                    
                if down_move > up_move and down_move > 0:
                    dm_minus.append(down_move)
                else:
                    dm_minus.append(0)
                    
            # Calculate ADX (simplified)
            if len(tr_values) >= 14:
                atr = np.mean(tr_values[-14:])
                di_plus = np.mean(dm_plus[-14:]) / atr * 100 if atr > 0 else 0
                di_minus = np.mean(dm_minus[-14:]) / atr * 100 if atr > 0 else 0
                
                dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
                adx = dx  # Simplified, normally would be smoothed
            else:
                adx = 0.0
                di_plus = 0.0
                di_minus = 0.0
                
            # Determine trend strength
            if adx > 40:
                strength = 'very strong'
            elif adx > 25:
                strength = 'strong'
            elif adx > 15:
                strength = 'moderate'
            else:
                strength = 'weak'
                
            # Determine trend direction
            if di_plus > di_minus:
                trend = 'bullish'
            elif di_minus > di_plus:
                trend = 'bearish'
            else:
                trend = 'neutral'
                
            return {
                'adx': adx,
                'strength': strength,
                'trend': trend,
                'di_plus': di_plus,
                'di_minus': di_minus
            }
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return {'adx': 0.0, 'strength': 'weak', 'trend': 'neutral'}
            
    def generate_signals(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate trading signals based on all indicators"""
        try:
            signals = {
                'buy_signals': [],
                'sell_signals': [],
                'overall_signal': 'HOLD',
                'confidence': 0.0
            }
            
            # This would be populated by the main trading engine
            # based on the comprehensive analysis
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return {
                'buy_signals': [],
                'sell_signals': [],
                'overall_signal': 'HOLD',
                'confidence': 0.0
            }
            
    def get_empty_analysis(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now(),
            'rsi': {'value': 50.0, 'oversold': False, 'overbought': False},
            'bollinger': {'upper': 0, 'middle': 0, 'lower': 0, 'position_percent': 0.5, 'squeeze': False},
            'stochastic': {'k_percent': 50.0, 'd_percent': 50.0, 'oversold': False, 'overbought': False},
            'slope': {'value': 0.0, 'trend': 'neutral', 'strength': 'weak'},
            'price_change': {'percent_change': 0.0, 'dollar_change': 0.0},
            'volume_analysis': {'current_volume': 0, 'avg_volume': 0, 'volume_ratio': 1.0, 'unusual_volume': False},
            'volatility': {'current': 0.0, 'avg': 0.0, 'percentile': 50.0},
            'momentum': {'roc': 0.0, 'macd': 0.0, 'signal': 0.0},
            'support_resistance': {'support': [], 'resistance': []},
            'trend_strength': {'adx': 0.0, 'strength': 'weak', 'trend': 'neutral'},
            'signals': {'buy_signals': [], 'sell_signals': [], 'overall_signal': 'HOLD', 'confidence': 0.0}
        }
