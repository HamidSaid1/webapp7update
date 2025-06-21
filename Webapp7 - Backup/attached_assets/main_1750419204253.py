"""
Interactive Brokers Trading Application - Enhanced Version
Main entry point for the trading application with sophisticated GUI and advanced market analysis.
"""

import tkinter as tk
import asyncio
import threading
import logging
import os
from datetime import datetime
from config import TradingConfig
from trading_engine import TradingEngine
from gui_components import TradingGUI
from logger_config import setup_logging
import nest_asyncio

# Enable nested asyncio for compatibility with tkinter
nest_asyncio.apply()

class TradingApplication:
    """Main application class that coordinates all components"""
    
    def __init__(self):
        self.setup_logging()
        self.config = TradingConfig()
        self.root = tk.Tk()
        self.trading_engine = TradingEngine(self.config)
        self.gui = TradingGUI(self.root, self.trading_engine, self.config)
        
        # Setup callbacks
        self.trading_engine.set_gui_callback(self.gui.update_market_data)
        self.trading_engine.set_trade_callback(self.gui.update_trades_display)
        self.trading_engine.set_status_callback(self.gui.update_status)
        
        self.logger = logging.getLogger(__name__)
        
        # Event loop for async operations
        self.event_loop = None
        
    def setup_logging(self):
        """Initialize logging configuration"""
        setup_logging(log_level=logging.INFO)
        
    def start_async_loop(self):
        """Start the asyncio event loop in a separate thread"""
        def run_loop():
            try:
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
                self.trading_engine.set_event_loop(self.event_loop)
                self.event_loop.run_forever()
            except Exception as e:
                self.logger.error(f"Async loop error: {e}")
                
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        self.logger.info("Async event loop started")
        
    def run(self):
        """Start the trading application"""
        try:
            self.logger.info("Starting Enhanced Trading Application")
            
            # Start async loop for IBKR operations
            self.start_async_loop()
            
            # Configure root window
            self.root.title("AutoTrade Plus - Enhanced Trading Platform")
            self.root.geometry("1600x1000")
            self.root.configure(bg="#f8f9fa")
            self.root.minsize(1200, 800)
            
            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start periodic tasks
            self.start_periodic_tasks()
            
            # Start GUI
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Application startup error: {e}")
            raise
    
    def start_periodic_tasks(self):
        """Start periodic background tasks"""
        def periodic_updates():
            try:
                # Update performance metrics
                self.trading_engine.update_performance_metrics()
                
                # Check for forced trade conditions
                self.trading_engine.check_force_trade_conditions()
                
                # Update cooldown timers
                self.trading_engine.update_cooldown_timers()
                
                # Schedule next update
                self.root.after(5000, periodic_updates)
                
            except Exception as e:
                self.logger.error(f"Periodic update error: {e}")
                self.root.after(5000, periodic_updates)
        
        # Start periodic updates
        self.root.after(1000, periodic_updates)
    
    def on_closing(self):
        """Handle application shutdown"""
        try:
            self.logger.info("Shutting down application")
            
            # Stop trading engine
            if self.trading_engine:
                self.trading_engine.stop_streaming()
                self.trading_engine.disconnect()
            
            # Stop event loop
            if self.event_loop and self.event_loop.is_running():
                self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            
            # Destroy GUI
            self.root.destroy()
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    # Ensure static directory exists
    os.makedirs("static", exist_ok=True)
    
    app = TradingApplication()
    app.run()
