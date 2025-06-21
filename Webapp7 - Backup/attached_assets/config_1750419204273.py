"""
Enhanced configuration management for the trading application
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List
import logging

@dataclass
class TradingConfig:
    """Enhanced configuration class for trading parameters"""
    
    # IBKR Connection Settings
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    client_id: int = 2
    timeout: int = 30
    
    # Default Trading Parameters
    default_tickers: str = "AAPL,MSFT,GOOGL,TSLA,NVDA"
    default_exchange: str = "SMART"
    equity_per_trade: float = 1000.0
    max_open_trades: int = 5
    hard_stop_loss: float = 0.03  # 3%
    take_profit: float = 0.06  # 6%
    
    # Enhanced Technical Analysis Parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_lower_threshold: float = 0.1  # 10% from lower band
    bb_upper_threshold: float = 0.9  # 90% from upper band
    
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    
    # Slope Analysis Parameters
    slope_period: int = 10
    slope_threshold: float = 0.02  # 2% slope threshold
    
    # Adaptive Threshold Parameters
    adaptive_lookback: int = 50
    adaptive_percentile: float = 0.8  # 80th percentile
    volatility_adjustment: float = 0.5
    
    # Spike Detection Parameters
    spike_threshold: float = 0.02  # 2% price spike
    spike_volume_multiplier: float = 2.0  # 2x average volume
    
    # Data Collection Settings
    data_points: int = 100
    chart_intervals: List[str] = field(default_factory=lambda: [
        "1 min", "3 mins", "5 mins", "10 mins", "15 mins", "30 mins", "1 hour"
    ])
    default_chart_interval: str = "5 mins"
    analysis_interval: int = 5  # seconds
    duration_str: str = "2 D"
    
    # Enhanced Risk Management
    max_daily_loss: float = 5000.0
    max_daily_trades: int = 20
    position_size_percent: float = 0.02  # 2% of account
    max_position_size: float = 10000.0
    cooldown_minutes: int = 30
    force_trade_cooldown: int = 60  # minutes
    
    # GUI Settings
    update_interval_ms: int = 1000
    theme_colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#007BFF',
        'success': '#28A745',
        'danger': '#DC3545',
        'warning': '#FFC107',
        'info': '#17A2B8',
        'light': '#F8F9FA',
        'dark': '#343A40'
    })
    
    # Banner Settings
    banner_enabled: bool = True
    banner_height: int = 80
    
    # Trading Hours (EST)
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.load_from_env()
        self.validate_config()
        
    def load_from_env(self):
        """Load configuration from environment variables"""
        # IBKR Settings
        self.ibkr_host = os.getenv("IBKR_HOST", self.ibkr_host)
        self.ibkr_port = int(os.getenv("IBKR_PORT", str(self.ibkr_port)))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", str(self.client_id)))
        
        # Trading Parameters
        self.equity_per_trade = float(os.getenv("EQUITY_PER_TRADE", str(self.equity_per_trade)))
        self.max_open_trades = int(os.getenv("MAX_OPEN_TRADES", str(self.max_open_trades)))
        self.hard_stop_loss = float(os.getenv("HARD_STOP_LOSS", str(self.hard_stop_loss)))
        self.take_profit = float(os.getenv("TAKE_PROFIT", str(self.take_profit)))
        
        # Technical Analysis
        self.rsi_period = int(os.getenv("RSI_PERIOD", str(self.rsi_period)))
        self.bb_period = int(os.getenv("BB_PERIOD", str(self.bb_period)))
        self.stoch_k_period = int(os.getenv("STOCH_K_PERIOD", str(self.stoch_k_period)))
        
        # Risk Management
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", str(self.max_daily_loss)))
        self.max_daily_trades = int(os.getenv("MAX_DAILY_TRADES", str(self.max_daily_trades)))
        self.position_size_percent = float(os.getenv("POSITION_SIZE_PERCENT", str(self.position_size_percent)))
        
    def validate_config(self):
        """Validate configuration parameters"""
        logger = logging.getLogger(__name__)
        
        validations = [
            (self.equity_per_trade > 0, "Equity per trade must be positive"),
            (self.max_open_trades > 0, "Max open trades must be positive"),
            (0 < self.hard_stop_loss < 1, "Hard stop loss must be between 0 and 1"),
            (0 < self.take_profit < 1, "Take profit must be between 0 and 1"),
            (self.rsi_period > 0, "RSI period must be positive"),
            (self.bb_period > 0, "Bollinger Bands period must be positive"),
            (self.stoch_k_period > 0, "Stochastic K period must be positive"),
            (0 < self.rsi_oversold < self.rsi_overbought < 100, "Invalid RSI thresholds"),
            (0 < self.stoch_oversold < self.stoch_overbought < 100, "Invalid Stochastic thresholds"),
            (self.max_daily_loss > 0, "Max daily loss must be positive"),
            (self.max_daily_trades > 0, "Max daily trades must be positive")
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(message)
        
        logger.info("Enhanced configuration validated successfully")
        
    def get_exchanges(self) -> List[str]:
        """Get available exchanges"""
        return ["SMART", "NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX"]
        
    def get_order_types(self) -> List[str]:
        """Get available order types"""
        return ["MKT", "LMT", "STP", "STP LMT", "TRAIL", "TRAIL LIMIT"]
        
    def get_time_in_force(self) -> List[str]:
        """Get available time in force options"""
        return ["DAY", "GTC", "IOC", "FOK"]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'ibkr_host': self.ibkr_host,
            'ibkr_port': self.ibkr_port,
            'client_id': self.client_id,
            'equity_per_trade': self.equity_per_trade,
            'max_open_trades': self.max_open_trades,
            'hard_stop_loss': self.hard_stop_loss,
            'take_profit': self.take_profit,
            'rsi_period': self.rsi_period,
            'bb_period': self.bb_period,
            'stoch_k_period': self.stoch_k_period,
            'max_daily_loss': self.max_daily_loss,
            'max_daily_trades': self.max_daily_trades,
            'cooldown_minutes': self.cooldown_minutes
        }
        
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update config from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate_config()
        
    def get_risk_parameters(self) -> Dict[str, float]:
        """Get risk management parameters"""
        return {
            'max_daily_loss': self.max_daily_loss,
            'max_daily_trades': self.max_daily_trades,
            'hard_stop_loss': self.hard_stop_loss,
            'take_profit': self.take_profit,
            'position_size_percent': self.position_size_percent,
            'max_position_size': self.max_position_size
        }
        
    def get_technical_parameters(self) -> Dict[str, Any]:
        """Get technical analysis parameters"""
        return {
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'bb_period': self.bb_period,
            'bb_std_dev': self.bb_std_dev,
            'stoch_k_period': self.stoch_k_period,
            'stoch_d_period': self.stoch_d_period,
            'stoch_oversold': self.stoch_oversold,
            'stoch_overbought': self.stoch_overbought,
            'slope_period': self.slope_period,
            'slope_threshold': self.slope_threshold
        }
