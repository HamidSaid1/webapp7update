"""
Enhanced logging configuration for the sophisticated trading application
Integrated from reserve_file.py with comprehensive logging capabilities
"""

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading

# Global logger registry
_loggers = {}
_log_lock = threading.Lock()

def setup_logging(log_level=logging.INFO, log_to_file=True, log_to_console=True, 
                 max_file_size=50*1024*1024, backup_count=10):
    """
    Setup comprehensive logging configuration with enhanced features
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        max_file_size: Maximum file size in bytes before rotation
        backup_count: Number of backup files to keep
    """
    
    with _log_lock:
        try:
            # Create logs directory if it doesn't exist
            if log_to_file:
                os.makedirs('logs', exist_ok=True)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            
            # Clear any existing handlers
            root_logger.handlers.clear()
            
            # Create enhanced formatter with more detail
            detailed_formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-25s | %(funcName)-20s:%(lineno)-4d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Simple formatter for console
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-5s | %(name)-15s | %(message)s',
                datefmt='%H:%M:%S'
            )
            
            # Console handler with color support
            if log_to_console:
                console_handler = EnhancedConsoleHandler()
                console_handler.setLevel(log_level)
                console_handler.setFormatter(console_formatter)
                root_logger.addHandler(console_handler)
            
            # File handlers
            if log_to_file:
                # Main application log
                main_log_file = f"logs/trading_app_{datetime.now().strftime('%Y%m%d')}.log"
                main_handler = logging.handlers.RotatingFileHandler(
                    main_log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                main_handler.setLevel(log_level)
                main_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(main_handler)
                
                # Error-only log
                error_log_file = f"logs/trading_errors_{datetime.now().strftime('%Y%m%d')}.log"
                error_handler = logging.handlers.RotatingFileHandler(
                    error_log_file,
                    maxBytes=max_file_size // 2,
                    backupCount=backup_count // 2,
                    encoding='utf-8'
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(error_handler)
                
                # Setup specialized loggers
                setup_specialized_loggers(detailed_formatter, max_file_size, backup_count)
            
            # Configure third-party library logging levels
            configure_third_party_loggers()
            
            # Install exception handler
            install_exception_handler()
            
            # Log startup message
            root_logger.info("="*100)
            root_logger.info("AutoTrade Plus - Enhanced Trading Application Started")
            root_logger.info(f"Log Level: {logging.getLevelName(log_level)}")
            root_logger.info(f"Console Logging: {'Enabled' if log_to_console else 'Disabled'}")
            root_logger.info(f"File Logging: {'Enabled' if log_to_file else 'Disabled'}")
            root_logger.info(f"Max File Size: {max_file_size:,} bytes")
            root_logger.info(f"Backup Count: {backup_count}")
            root_logger.info("="*100)
            
        except Exception as e:
            print(f"Logging setup error: {e}")
            traceback.print_exc()

def setup_specialized_loggers(formatter, max_file_size, backup_count):
    """Setup specialized loggers for different components"""
    try:
        specialized_loggers = {
            'trading_activity': {
                'filename': 'trading_activity',
                'level': logging.INFO,
                'propagate': False,
                'max_size': max_file_size
            },
            'market_data': {
                'filename': 'market_data',
                'level': logging.DEBUG,
                'propagate': False,
                'max_size': max_file_size // 2
            },
            'risk_management': {
                'filename': 'risk_management',
                'level': logging.WARNING,
                'propagate': False,
                'max_size': max_file_size // 4
            },
            'performance': {
                'filename': 'performance',
                'level': logging.INFO,
                'propagate': False,
                'max_size': max_file_size // 4
            },
            'connection': {
                'filename': 'connection',
                'level': logging.INFO,
                'propagate': False,
                'max_size': max_file_size // 8
            },
            'technical_analysis': {
                'filename': 'technical_analysis',
                'level': logging.DEBUG,
                'propagate': False,
                'max_size': max_file_size // 4
            }
        }
        
        for logger_name, config in specialized_loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(config['level'])
            logger.propagate = config['propagate']
            
            # Create file handler
            log_file = f"logs/{config['filename']}_{datetime.now().strftime('%Y%m%d')}.log"
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=config['max_size'],
                backupCount=backup_count,
                encoding='utf-8'
            )
            handler.setLevel(config['level'])
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            _loggers[logger_name] = logger
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Specialized logger setup error: {e}")

def configure_third_party_loggers():
    """Configure logging levels for third-party libraries"""
    try:
        third_party_levels = {
            'ib_insync': logging.WARNING,
            'urllib3': logging.WARNING,
            'PIL': logging.WARNING,
            'matplotlib': logging.WARNING,
            'pandas': logging.WARNING,
            'numpy': logging.WARNING,
            'requests': logging.WARNING,
            'asyncio': logging.WARNING
        }
        
        for library, level in third_party_levels.items():
            logging.getLogger(library).setLevel(level)
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Third-party logger configuration error: {e}")

class EnhancedConsoleHandler(logging.StreamHandler):
    """Enhanced console handler with color support"""
    
    COLOR_CODES = {
        logging.DEBUG: '\033[36m',      # Cyan
        logging.INFO: '\033[32m',       # Green
        logging.WARNING: '\033[33m',    # Yellow
        logging.ERROR: '\033[31m',      # Red
        logging.CRITICAL: '\033[35m'    # Magenta
    }
    RESET_CODE = '\033[0m'
    
    def emit(self, record):
        try:
            # Add color codes for terminal output
            if hasattr(self.stream, 'isatty') and self.stream.isatty():
                color_code = self.COLOR_CODES.get(record.levelno, '')
                record.levelname = f"{color_code}{record.levelname}{self.RESET_CODE}"
                
            super().emit(record)
            
        except Exception as e:
            self.handleError(record)

def install_exception_handler():
    """Install global exception handler"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow Ctrl+C to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = logging.getLogger('exception_handler')
        logger.critical(
            "Uncaught exception occurred",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    # Set the exception handler
    sys.excepthook = handle_exception

# Specialized logging functions

def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return _loggers.get(name, logging.getLogger(name))

def log_trade_event(event_type: str, symbol: str, details: Dict[str, Any]):
    """
    Log trading events with structured format
    
    Args:
        event_type: Type of event (BUY, SELL, ERROR, etc.)
        symbol: Stock symbol
        details: Dictionary of additional details
    """
    try:
        trade_logger = get_logger('trading_activity')
        
        # Format the log message
        message_parts = [f"[{event_type}]", f"Symbol: {symbol}"]
        
        for key, value in details.items():
            if isinstance(value, float):
                if 'price' in key.lower() or 'pnl' in key.lower() or 'amount' in key.lower():
                    message_parts.append(f"{key}: ${value:.2f}")
                elif 'percent' in key.lower() or 'rate' in key.lower():
                    message_parts.append(f"{key}: {value:.2f}%")
                else:
                    message_parts.append(f"{key}: {value:.4f}")
            elif isinstance(value, datetime):
                message_parts.append(f"{key}: {value.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                message_parts.append(f"{key}: {value}")
        
        log_message = " | ".join(message_parts)
        trade_logger.info(log_message)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Trade event logging error: {e}")

def log_connection_event(event_type: str, details: str):
    """Log IBKR connection events"""
    try:
        logger = get_logger('connection')
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        logger.info(f"[{timestamp}] [{event_type}] {details}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Connection event logging error: {e}")

def log_market_data_event(symbol: str, data: Dict[str, Any]):
    """Log market data updates"""
    try:
        logger = get_logger('market_data')
        
        # Extract key market data
        price = data.get('last', 0)
        volume = data.get('volume', 0)
        timestamp = data.get('timestamp', datetime.now())
        
        logger.debug(f"Market Data | {symbol} | Price: ${price:.2f} | Volume: {volume:,} | Time: {timestamp}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Market data logging error: {e}")

def log_analysis_event(symbol: str, analysis_results: Dict[str, Any]):
    """Log technical analysis results"""
    try:
        logger = get_logger('technical_analysis')
        
        # Extract key metrics for logging
        rsi = analysis_results.get('rsi', {}).get('value', 0)
        bb_pos = analysis_results.get('bollinger', {}).get('position_percent', 0)
        stoch_k = analysis_results.get('stochastic', {}).get('k_percent', 0)
        slope = analysis_results.get('slope', {}).get('value', 0)
        signal = analysis_results.get('signals', {}).get('overall_signal', 'HOLD')
        
        logger.debug(
            f"Analysis | {symbol} | RSI: {rsi:.1f} | BB%: {bb_pos:.1f} | "
            f"Stoch: {stoch_k:.1f} | Slope: {slope:.3f} | Signal: {signal}"
        )
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Analysis event logging error: {e}")

def log_risk_event(event_type: str, symbol: str, risk_details: Dict[str, Any]):
    """Log risk management events"""
    try:
        logger = get_logger('risk_management')
        
        message_parts = [f"[{event_type}]", f"Symbol: {symbol}"]
        
        for key, value in risk_details.items():
            if isinstance(value, float):
                if 'amount' in key.lower() or 'loss' in key.lower() or 'profit' in key.lower():
                    message_parts.append(f"{key}: ${value:.2f}")
                elif 'percent' in key.lower() or 'ratio' in key.lower():
                    message_parts.append(f"{key}: {value:.2f}%")
                else:
                    message_parts.append(f"{key}: {value}")
            else:
                message_parts.append(f"{key}: {value}")
        
        log_message = " | ".join(message_parts)
        logger.warning(log_message)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Risk event logging error: {e}")

def log_performance_metrics(metrics: Dict[str, Any]):
    """Log performance metrics periodically"""
    try:
        logger = get_logger('performance')
        
        message_parts = ["PERFORMANCE SNAPSHOT"]
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'pnl' in key.lower() or 'balance' in key.lower() or 'profit' in key.lower():
                    message_parts.append(f"{key}: ${value:.2f}")
                elif 'rate' in key.lower() or 'percent' in key.lower() or 'ratio' in key.lower():
                    message_parts.append(f"{key}: {value:.2f}%")
                else:
                    message_parts.append(f"{key}: {value:.4f}")
            else:
                message_parts.append(f"{key}: {value}")
        
        log_message = " | ".join(message_parts)
        logger.info(log_message)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Performance metrics logging error: {e}")

def log_gui_event(event_type: str, details: str):
    """Log GUI events and user interactions"""
    try:
        logger = logging.getLogger('gui')
        logger.info(f"[GUI] [{event_type}] {details}")
    except Exception as e:
        logging.getLogger(__name__).error(f"GUI event logging error: {e}")

def log_system_event(event_type: str, details: Dict[str, Any]):
    """Log system-level events"""
    try:
        logger = logging.getLogger('system')
        
        message = f"[{event_type}]"
        for key, value in details.items():
            message += f" | {key}: {value}"
            
        logger.info(message)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"System event logging error: {e}")

def setup_debug_logging():
    """Setup debug-level logging for development"""
    try:
        setup_logging(log_level=logging.DEBUG, log_to_file=True, log_to_console=True)
        
        # Additional debug loggers
        debug_loggers = [
            'trading_engine',
            'technical_analyzer', 
            'risk_manager',
            'gui_components',
            'market_data_processor',
            'order_manager'
        ]
        
        for logger_name in debug_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            
        logging.getLogger(__name__).info("Debug logging enabled for development")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Debug logging setup error: {e}")

def cleanup_old_logs(days_to_keep: int = 30):
    """Clean up log files older than specified days"""
    try:
        import glob
        from pathlib import Path
        
        if not os.path.exists('logs'):
            return
            
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        log_files = glob.glob('logs/*.log*')
        cleaned_count = 0
        
        for log_file in log_files:
            try:
                file_path = Path(log_file)
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    os.remove(log_file)
                    cleaned_count += 1
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not remove log file {log_file}: {e}")
                
        if cleaned_count > 0:
            logging.getLogger(__name__).info(f"Cleaned up {cleaned_count} old log files")
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Log cleanup error: {e}")

def get_log_statistics() -> Dict[str, Any]:
    """Get logging statistics"""
    try:
        stats = {
            'log_files': [],
            'total_size': 0,
            'oldest_log': None,
            'newest_log': None
        }
        
        if not os.path.exists('logs'):
            return stats
            
        log_files = []
        for filename in os.listdir('logs'):
            if filename.endswith('.log'):
                filepath = os.path.join('logs', filename)
                file_stat = os.stat(filepath)
                
                log_files.append({
                    'name': filename,
                    'size': file_stat.st_size,
                    'modified': datetime.fromtimestamp(file_stat.st_mtime)
                })
                
                stats['total_size'] += file_stat.st_size
        
        if log_files:
            log_files.sort(key=lambda x: x['modified'])
            stats['oldest_log'] = log_files[0]['modified']
            stats['newest_log'] = log_files[-1]['modified']
            stats['log_files'] = log_files
            
        return stats
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Log statistics error: {e}")
        return {}

# Context managers for structured logging

class LogContext:
    """Context manager for structured logging with timing"""
    
    def __init__(self, logger_name: str, operation: str, **kwargs):
        self.logger = get_logger(logger_name)
        self.operation = operation
        self.context_data = kwargs
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}", extra=self.context_data)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {duration.total_seconds():.3f}s",
                extra=self.context_data
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration.total_seconds():.3f}s: {exc_val}",
                extra=self.context_data
            )
            
        return False  # Don't suppress exceptions

def log_with_context(logger_name: str, operation: str, **kwargs):
    """Decorator for logging function execution with context"""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            with LogContext(logger_name, operation, **kwargs):
                return func(*args, **func_kwargs)
        return wrapper
    return decorator

# Performance logging utilities

class PerformanceLogger:
    """Performance logger for monitoring system performance"""
    
    def __init__(self):
        self.logger = get_logger('performance')
        self.metrics = {}
        
    def record_metric(self, metric_name: str, value: float, unit: str = ""):
        """Record a performance metric"""
        timestamp = datetime.now()
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            
        self.metrics[metric_name].append({
            'timestamp': timestamp,
            'value': value,
            'unit': unit
        })
        
        self.logger.debug(f"Performance | {metric_name}: {value} {unit}")
        
    def log_summary(self):
        """Log performance summary"""
        try:
            summary = {}
            
            for metric_name, readings in self.metrics.items():
                if readings:
                    values = [r['value'] for r in readings]
                    summary[metric_name] = {
                        'count': len(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'latest': values[-1],
                        'unit': readings[-1]['unit']
                    }
            
            self.logger.info(f"Performance Summary: {summary}")
            
        except Exception as e:
            self.logger.error(f"Performance summary error: {e}")

# Global performance logger instance
performance_logger = PerformanceLogger()
