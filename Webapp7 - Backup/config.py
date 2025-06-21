"""
Enhanced configuration management for the trading application
Exact parameters from reserve_file.py
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List
import logging

@dataclass
class TradingConfig:
    """Enhanced configuration class for trading parameters from reserve_file.py"""
    
    # IBKR Connection Settings (TWS Paper Trading)
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497  # TWS paper trading port
    client_id: int = 1
    timeout: int = 10
    paper_trading: bool = True
    
    # Default Trading Parameters (exact from reserve_file.py)
    default_tickers: str = "ABTS"
    default_exchange: str = "SMART"
    equity_per_trade: float = 100.0
    max_open_trades: int = 2
    hard_stop_loss: float = 0.03  # 3%
    
    # Technical Analysis Parameters (exact from reserve_file.py)
    rsi_period: int = 2
    bb_period: int = 2
    stoch_k_period: int = 2
    stoch_d_period: int = 2
    stoch_k_min: float = 50.0
    stoch_d_min: float = 40.0
    
    # Data Collection Settings (exact from reserve_file.py)
    data_points: int = 15
    chart_intervals: List[str] = field(default_factory=lambda: [
        "5 sec", "10 sec", "30 sec", "1 min", "3 min", "5 min", "10 min", "15 min"
    ])
    default_chart_interval: str = "5 mins"
    analysis_interval: int = 1
    duration_str: str = "1 D"
    
    # GUI Settings (from reserve_file.py)
    update_interval_ms: int = 5000
    theme_colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#007BFF',
        'success': '#28A745',
        'danger': '#DC3545',
        'warning': '#FFC107',
        'info': '#17A2B8',
        'light': '#F8F9FA',
        'dark': '#343A40',
        'white': '#FFFFFF'
    })
    
    # Original trading criteria thresholds (from reserve_file.py)
    price_rate_threshold: float = 0.257142857
    rsi_rate_threshold: float = 0.201904762
    bollinger_rate_threshold: float = 1.48571429
    
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
        
    def validate_config(self):
        """Validate configuration parameters"""
        logger = logging.getLogger(__name__)
        
        validations = [
            (self.equity_per_trade > 0, "Equity per trade must be positive"),
            (self.max_open_trades > 0, "Max open trades must be positive"),
            (0 < self.hard_stop_loss < 1, "Hard stop loss must be between 0 and 1"),
            (self.rsi_period > 0, "RSI period must be positive"),
            (self.bb_period > 0, "Bollinger Bands period must be positive"),
            (self.stoch_k_period > 0, "Stochastic K period must be positive")
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(message)
        
        logger.info("Trading configuration validated successfully")
        
    def get_exchanges(self) -> List[str]:
        """Get available exchanges (from reserve_file.py)"""
        return ["SMART", "NYSE", "NASDAQ"]
        
    def get_chart_intervals(self) -> List[str]:
        """Get available chart intervals (from reserve_file.py)"""
        return ["1 min", "3 min", "5 min", "15 min"]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'ibkr_host': self.ibkr_host,
            'ibkr_port': self.ibkr_port,
            'client_id': self.client_id,
            'equity_per_trade': self.equity_per_trade,
            'max_open_trades': self.max_open_trades,
            'hard_stop_loss': self.hard_stop_loss,
            'rsi_period': self.rsi_period,
            'bb_period': self.bb_period,
            'stoch_k_period': self.stoch_k_period,
            'stoch_d_period': self.stoch_d_period,
            'stoch_k_min': self.stoch_k_min,
            'stoch_d_min': self.stoch_d_min
        }
