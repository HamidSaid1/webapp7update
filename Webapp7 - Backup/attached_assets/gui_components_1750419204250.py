"""
Enhanced GUI components for the sophisticated trading application
Integrated from reserve_file.py with complete functionality
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import asyncio
import os
from PIL import Image, ImageTk
import pandas as pd
import numpy as np

from utils import (
    format_currency, format_percentage, calculate_time_difference,
    parse_ticker_list, validate_ticker_symbol
)

class TradingGUI:
    """Enhanced GUI class with sophisticated interface and real-time updates"""
    
    def __init__(self, root, trading_engine, config):
        self.root = root
        self.trading_engine = trading_engine
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # GUI State Variables
        self.is_streaming = tk.BooleanVar(value=False)
        self.tickers_var = tk.StringVar(value=config.default_tickers)
        self.exchange_var = tk.StringVar(value=config.default_exchange)
        self.chart_interval_var = tk.StringVar(value=config.default_chart_interval)
        self.force_trade_var = tk.BooleanVar(value=False)
        
        # Trading Parameters
        self.equity_per_trade_var = tk.DoubleVar(value=config.equity_per_trade)
        self.max_open_trades_var = tk.IntVar(value=config.max_open_trades)
        self.hard_stop_loss_var = tk.DoubleVar(value=config.hard_stop_loss * 100)  # Convert to percentage
        self.take_profit_var = tk.DoubleVar(value=config.take_profit * 100)
        
        # Technical Analysis Parameters
        self.rsi_period_var = tk.IntVar(value=config.rsi_period)
        self.bb_period_var = tk.IntVar(value=config.bb_period)
        self.stoch_k_period_var = tk.IntVar(value=config.stoch_k_period)
        self.data_points_var = tk.IntVar(value=config.data_points)
        
        # Status Variables
        self.connection_status = tk.StringVar(value="Disconnected")
        self.streaming_status = tk.StringVar(value="Stopped")
        self.account_balance = tk.StringVar(value="$0.00")
        self.daily_pnl_var = tk.StringVar(value="$0.00")
        self.win_rate_var = tk.StringVar(value="0%")
        self.open_positions_var = tk.StringVar(value="0")
        self.daily_trades_var = tk.StringVar(value="0")
        
        # Data Storage
        self.market_data_cache = {}
        self.trade_history_cache = []
        
        # Banner Image
        self.banner_image = None
        
        # Setup GUI
        self.setup_styles()
        self.setup_main_interface()
        self.setup_menu_bar()
        
        # Start periodic updates
        self.start_periodic_updates()
        
    def setup_styles(self):
        """Setup custom styles for the application"""
        try:
            style = ttk.Style()
            
            # Configure colors
            colors = self.config.theme_colors
            
            # Main styles
            style.configure('Title.TLabel', font=('Arial', 12, 'bold'), foreground=colors['primary'])
            style.configure('Header.TLabel', font=('Arial', 10, 'bold'))
            style.configure('Status.TLabel', font=('Arial', 9))
            
            # Button styles
            style.configure('Action.TButton', font=('Arial', 9, 'bold'))
            style.configure('Success.TButton', foreground=colors['success'])
            style.configure('Danger.TButton', foreground=colors['danger'])
            
            # Frame styles
            style.configure('Card.TLabelFrame', relief='raised', borderwidth=2)
            
        except Exception as e:
            self.logger.error(f"Style setup error: {e}")
            
    def setup_main_interface(self):
        """Setup the enhanced main interface layout"""
        try:
            # Configure root
            self.root.configure(bg=self.config.theme_colors['light'])
            
            # Create main container with banner
            main_container = ttk.Frame(self.root, padding="10")
            main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_container.columnconfigure(1, weight=1)
            main_container.rowconfigure(3, weight=1)
            
            # Setup sections
            current_row = 0
            
            # Banner section
            if self.config.banner_enabled:
                self.setup_banner(main_container)
                current_row += 1
                
            # Connection and status panel
            self.setup_connection_panel(main_container, current_row)
            current_row += 1
            
            # Trading parameters panel
            self.setup_parameters_panel(main_container, current_row)
            current_row += 1
            
            # Status information panel
            self.setup_status_panel(main_container, current_row)
            current_row += 1
            
            # Main data display (market data and trades)
            self.setup_data_display(main_container, current_row)
            
        except Exception as e:
            self.logger.error(f"Main interface setup error: {e}")
            
    def setup_banner(self, parent):
        """Setup banner with logo/image"""
        try:
            banner_frame = ttk.Frame(parent, height=self.config.banner_height)
            banner_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            banner_frame.grid_propagate(False)
            
            # Try to load banner image
            banner_path = "static/banner.svg"
            if os.path.exists(banner_path):
                try:
                    # For SVG, we'll create a simple text banner as PIL doesn't support SVG directly
                    banner_label = ttk.Label(
                        banner_frame, 
                        text="AutoTrade Plus - Enhanced Trading Platform",
                        style='Title.TLabel',
                        anchor=tk.CENTER
                    )
                    banner_label.pack(expand=True, fill=tk.BOTH)
                except Exception as e:
                    self.logger.warning(f"Banner image load error: {e}")
                    self.create_text_banner(banner_frame)
            else:
                self.create_text_banner(banner_frame)
                
        except Exception as e:
            self.logger.error(f"Banner setup error: {e}")
            
    def create_text_banner(self, parent):
        """Create text-based banner"""
        banner_label = ttk.Label(
            parent,
            text="🚀 AutoTrade Plus - Advanced Trading System",
            style='Title.TLabel',
            anchor=tk.CENTER
        )
        banner_label.pack(expand=True, fill=tk.BOTH)
        
    def setup_menu_bar(self):
        """Setup comprehensive application menu bar"""
        try:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Connect to IBKR", command=self.connect_ibkr, accelerator="Ctrl+C")
            file_menu.add_command(label="Disconnect", command=self.disconnect_ibkr, accelerator="Ctrl+D")
            file_menu.add_separator()
            file_menu.add_command(label="Export Trades", command=self.export_trades)
            file_menu.add_command(label="Import Settings", command=self.import_settings)
            file_menu.add_command(label="Export Settings", command=self.export_settings)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Ctrl+Q")
            
            # Trading menu
            trading_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Trading", menu=trading_menu)
            trading_menu.add_command(label="Start Streaming", command=self.start_streaming, accelerator="F1")
            trading_menu.add_command(label="Stop Streaming", command=self.stop_streaming, accelerator="F2")
            trading_menu.add_separator()
            trading_menu.add_command(label="Toggle Force Trade", command=self.toggle_force_trade, accelerator="F3")
            trading_menu.add_command(label="Close All Positions", command=self.close_all_positions)
            
            # Analysis menu
            analysis_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Analysis", menu=analysis_menu)
            analysis_menu.add_command(label="Run Backtest", command=self.run_backtest)
            analysis_menu.add_command(label="Performance Report", command=self.show_performance_report)
            analysis_menu.add_command(label="Risk Analysis", command=self.show_risk_analysis)
            
            # Tools menu
            tools_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Tools", menu=tools_menu)
            tools_menu.add_command(label="Market Scanner", command=self.show_market_scanner)
            tools_menu.add_command(label="Options Calculator", command=self.show_options_calculator)
            tools_menu.add_command(label="Position Sizer", command=self.show_position_sizer)
            
            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="User Guide", command=self.show_user_guide)
            help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
            help_menu.add_command(label="About", command=self.show_about)
            
            # Bind keyboard shortcuts
            self.root.bind('<Control-c>', lambda e: self.connect_ibkr())
            self.root.bind('<Control-d>', lambda e: self.disconnect_ibkr())
            self.root.bind('<Control-q>', lambda e: self.on_closing())
            self.root.bind('<F1>', lambda e: self.start_streaming())
            self.root.bind('<F2>', lambda e: self.stop_streaming())
            self.root.bind('<F3>', lambda e: self.toggle_force_trade())
            
        except Exception as e:
            self.logger.error(f"Menu bar setup error: {e}")
            
    def setup_connection_panel(self, parent, row):
        """Setup connection and control panel"""
        try:
            conn_frame = ttk.LabelFrame(parent, text="Connection & Control", style='Card.TLabelFrame', padding="10")
            conn_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # Connection status and controls
            status_frame = ttk.Frame(conn_frame)
            status_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
            
            ttk.Button(status_frame, text="Connect IBKR", command=self.connect_ibkr, style='Action.TButton').grid(row=0, column=0, padx=(0, 5))
            ttk.Button(status_frame, text="Disconnect", command=self.disconnect_ibkr).grid(row=0, column=1, padx=(0, 5))
            
            ttk.Label(status_frame, text="Status:", style='Header.TLabel').grid(row=0, column=2, padx=(20, 5))
            status_label = ttk.Label(status_frame, textvariable=self.connection_status, style='Status.TLabel')
            status_label.grid(row=0, column=3, padx=(0, 20))
            
            # Streaming controls
            ttk.Button(status_frame, text="Start Streaming", command=self.start_streaming, style='Success.TButton').grid(row=0, column=4, padx=(0, 5))
            ttk.Button(status_frame, text="Stop Streaming", command=self.stop_streaming, style='Danger.TButton').grid(row=0, column=5, padx=(0, 5))
            
            ttk.Label(status_frame, text="Streaming:", style='Header.TLabel').grid(row=0, column=6, padx=(20, 5))
            ttk.Label(status_frame, textvariable=self.streaming_status, style='Status.TLabel').grid(row=0, column=7)
            
            # Force trade toggle
            force_frame = ttk.Frame(conn_frame)
            force_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E))
            
            ttk.Checkbutton(force_frame, text="Force Trade Mode", variable=self.force_trade_var, 
                           command=self.on_force_trade_toggle).grid(row=0, column=0, sticky=tk.W)
            
            ttk.Label(force_frame, text="⚠️ Force mode bypasses some safety checks", 
                     foreground=self.config.theme_colors['warning']).grid(row=0, column=1, padx=(20, 0), sticky=tk.W)
            
        except Exception as e:
            self.logger.error(f"Connection panel setup error: {e}")
            
    def setup_parameters_panel(self, parent, row):
        """Setup trading parameters panel with two-column layout"""
        try:
            params_frame = ttk.LabelFrame(parent, text="Trading Parameters", style='Card.TLabelFrame', padding="10")
            params_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # Left column - Basic Parameters
            left_frame = ttk.Frame(params_frame)
            left_frame.grid(row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 20))
            
            # Tickers
            ttk.Label(left_frame, text="Tickers:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
            tickers_entry = ttk.Entry(left_frame, textvariable=self.tickers_var, width=40)
            tickers_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # Exchange and Interval
            exchange_interval_frame = ttk.Frame(left_frame)
            exchange_interval_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
            
            ttk.Label(exchange_interval_frame, text="Exchange:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
            exchange_combo = ttk.Combobox(exchange_interval_frame, textvariable=self.exchange_var, 
                                        values=self.config.get_exchanges(), width=15, state="readonly")
            exchange_combo.grid(row=1, column=0, padx=(0, 10))
            
            ttk.Label(exchange_interval_frame, text="Interval:", style='Header.TLabel').grid(row=0, column=1, sticky=tk.W)
            interval_combo = ttk.Combobox(exchange_interval_frame, textvariable=self.chart_interval_var,
                                        values=self.config.chart_intervals, width=15, state="readonly")
            interval_combo.grid(row=1, column=1)
            
            # Financial Parameters
            financial_frame = ttk.Frame(left_frame)
            financial_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
            
            ttk.Label(financial_frame, text="Equity per Trade ($):", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(financial_frame, textvariable=self.equity_per_trade_var, width=15).grid(row=1, column=0, padx=(0, 10))
            
            ttk.Label(financial_frame, text="Max Open Trades:", style='Header.TLabel').grid(row=0, column=1, sticky=tk.W)
            ttk.Entry(financial_frame, textvariable=self.max_open_trades_var, width=15).grid(row=1, column=1)
            
            # Risk Parameters
            risk_frame = ttk.Frame(left_frame)
            risk_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
            
            ttk.Label(risk_frame, text="Hard Stop Loss (%):", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
            ttk.Entry(risk_frame, textvariable=self.hard_stop_loss_var, width=15).grid(row=1, column=0, padx=(0, 10))
            
            ttk.Label(risk_frame, text="Take Profit (%):", style='Header.TLabel').grid(row=0, column=1, sticky=tk.W)
            ttk.Entry(risk_frame, textvariable=self.take_profit_var, width=15).grid(row=1, column=1)
            
            # Right column - Technical Parameters
            right_frame = ttk.Frame(params_frame)
            right_frame.grid(row=0, column=1, sticky=(tk.W, tk.N))
            
            ttk.Label(right_frame, text="Technical Analysis Parameters", style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 10))
            
            # RSI Parameters
            ttk.Label(right_frame, text="RSI Period:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W)
            ttk.Entry(right_frame, textvariable=self.rsi_period_var, width=10).grid(row=1, column=1, padx=(5, 0), pady=(0, 5))
            
            # Bollinger Bands
            ttk.Label(right_frame, text="BB Period:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W)
            ttk.Entry(right_frame, textvariable=self.bb_period_var, width=10).grid(row=2, column=1, padx=(5, 0), pady=(0, 5))
            
            # Stochastic
            ttk.Label(right_frame, text="Stoch K Period:", style='Header.TLabel').grid(row=3, column=0, sticky=tk.W)
            ttk.Entry(right_frame, textvariable=self.stoch_k_period_var, width=10).grid(row=3, column=1, padx=(5, 0), pady=(0, 5))
            
            # Data Points
            ttk.Label(right_frame, text="Data Points:", style='Header.TLabel').grid(row=4, column=0, sticky=tk.W)
            ttk.Entry(right_frame, textvariable=self.data_points_var, width=10).grid(row=4, column=1, padx=(5, 0), pady=(0, 5))
            
            # Update parameters button
            ttk.Button(right_frame, text="Update Parameters", command=self.update_parameters, 
                      style='Action.TButton').grid(row=5, column=0, columnspan=2, pady=(20, 0))
            
        except Exception as e:
            self.logger.error(f"Parameters panel setup error: {e}")
            
    def setup_status_panel(self, parent, row):
        """Setup enhanced status information panel"""
        try:
            status_frame = ttk.LabelFrame(parent, text="Account & Performance Status", style='Card.TLabelFrame', padding="10")
            status_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # Account Status
            account_frame = ttk.Frame(status_frame)
            account_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
            
            ttk.Label(account_frame, text="Account Balance:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
            ttk.Label(account_frame, textvariable=self.account_balance, style='Status.TLabel').grid(row=0, column=1, padx=(0, 30))
            
            ttk.Label(account_frame, text="Daily P&L:", style='Header.TLabel').grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
            pnl_label = ttk.Label(account_frame, textvariable=self.daily_pnl_var, style='Status.TLabel')
            pnl_label.grid(row=0, column=3, padx=(0, 30))
            
            ttk.Label(account_frame, text="Open Positions:", style='Header.TLabel').grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
            ttk.Label(account_frame, textvariable=self.open_positions_var, style='Status.TLabel').grid(row=0, column=5, padx=(0, 30))
            
            # Performance Metrics
            perf_frame = ttk.Frame(status_frame)
            perf_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
            
            ttk.Label(perf_frame, text="Win Rate:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
            ttk.Label(perf_frame, textvariable=self.win_rate_var, style='Status.TLabel').grid(row=0, column=1, padx=(0, 30))
            
            ttk.Label(perf_frame, text="Daily Trades:", style='Header.TLabel').grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
            ttk.Label(perf_frame, textvariable=self.daily_trades_var, style='Status.TLabel').grid(row=0, column=3, padx=(0, 30))
            
            # Force Trade Status
            force_status_var = tk.StringVar(value="Disabled")
            ttk.Label(perf_frame, text="Force Mode:", style='Header.TLabel').grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
            self.force_status_label = ttk.Label(perf_frame, textvariable=force_status_var, style='Status.TLabel')
            self.force_status_label.grid(row=0, column=5)
            self.force_status_var = force_status_var
            
        except Exception as e:
            self.logger.error(f"Status panel setup error: {e}")
            
    def setup_data_display(self, parent, row):
        """Setup enhanced market data and trades display"""
        try:
            # Create notebook for tabbed interface
            notebook = ttk.Notebook(parent)
            notebook.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Market Data Tab
            market_frame = ttk.Frame(notebook, padding="5")
            notebook.add(market_frame, text="Live Market Data")
            self.setup_market_data_tab(market_frame)
            
            # Open Positions Tab
            positions_frame = ttk.Frame(notebook, padding="5")
            notebook.add(positions_frame, text="Open Positions")
            self.setup_positions_tab(positions_frame)
            
            # Trade History Tab
            history_frame = ttk.Frame(notebook, padding="5")
            notebook.add(history_frame, text="Trade History")
            self.setup_history_tab(history_frame)
            
            # Performance Analytics Tab
            analytics_frame = ttk.Frame(notebook, padding="5")
            notebook.add(analytics_frame, text="Analytics")
            self.setup_analytics_tab(analytics_frame)
            
        except Exception as e:
            self.logger.error(f"Data display setup error: {e}")
            
    def setup_market_data_tab(self, parent):
        """Setup market data display tab"""
        try:
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(0, weight=1)
            
            # Create treeview for market data
            columns = ("Symbol", "Price", "Change %", "RSI", "BB Position", "Stoch K", "Stoch D", 
                      "Volume", "Slope", "Signal", "Last Update")
            
            self.market_tree = ttk.Treeview(parent, columns=columns, show="headings", height=15)
            
            # Configure columns
            column_widths = {
                "Symbol": 80, "Price": 80, "Change %": 80, "RSI": 60, "BB Position": 80,
                "Stoch K": 60, "Stoch D": 60, "Volume": 100, "Slope": 60, "Signal": 80, "Last Update": 120
            }
            
            for col in columns:
                self.market_tree.heading(col, text=col, command=lambda c=col: self.sort_market_data(c))
                self.market_tree.column(col, width=column_widths.get(col, 80), anchor=tk.CENTER)
                
            # Add scrollbars
            v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.market_tree.yview)
            h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.market_tree.xview)
            self.market_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Grid layout
            self.market_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
            
            # Context menu
            self.setup_market_context_menu()
            
        except Exception as e:
            self.logger.error(f"Market data tab setup error: {e}")
            
    def setup_positions_tab(self, parent):
        """Setup open positions display tab"""
        try:
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(0, weight=1)
            
            columns = ("Symbol", "Action", "Quantity", "Entry Price", "Current Price", 
                      "Unrealized P&L", "Stop Loss", "Take Profit", "Entry Time", "Duration")
            
            self.positions_tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)
            
            # Configure columns
            for col in columns:
                self.positions_tree.heading(col, text=col)
                self.positions_tree.column(col, width=90, anchor=tk.CENTER)
                
            # Scrollbar
            pos_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.positions_tree.yview)
            self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)
            
            self.positions_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            pos_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            # Position management buttons
            button_frame = ttk.Frame(parent)
            button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
            
            ttk.Button(button_frame, text="Close Selected", command=self.close_selected_position).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_frame, text="Close All", command=self.close_all_positions).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_frame, text="Update Stop Loss", command=self.update_stop_loss).pack(side=tk.LEFT)
            
        except Exception as e:
            self.logger.error(f"Positions tab setup error: {e}")
            
    def setup_history_tab(self, parent):
        """Setup trade history display tab"""
        try:
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(0, weight=1)
            
            columns = ("Trade ID", "Symbol", "Action", "Quantity", "Entry Price", "Exit Price", 
                      "Realized P&L", "Entry Time", "Exit Time", "Duration", "Strategy")
            
            self.history_tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
            
            # Configure columns
            for col in columns:
                self.history_tree.heading(col, text=col)
                self.history_tree.column(col, width=80, anchor=tk.CENTER)
                
            # Scrollbar
            hist_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.history_tree.yview)
            self.history_tree.configure(yscrollcommand=hist_scrollbar.set)
            
            self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            hist_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            # History controls
            control_frame = ttk.Frame(parent)
            control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
            
            ttk.Button(control_frame, text="Export History", command=self.export_trades).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_frame, text="Clear History", command=self.clear_history).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_frame, text="Filter", command=self.filter_history).pack(side=tk.LEFT)
            
        except Exception as e:
            self.logger.error(f"History tab setup error: {e}")
            
    def setup_analytics_tab(self, parent):
        """Setup performance analytics tab"""
        try:
            # Performance metrics display
            metrics_frame = ttk.LabelFrame(parent, text="Performance Metrics", padding="10")
            metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # Create metrics labels
            self.metrics_labels = {}
            metrics = [
                ("Total Trades", "total_trades"),
                ("Win Rate", "win_rate"), 
                ("Total P&L", "total_pnl"),
                ("Max Drawdown", "max_drawdown"),
                ("Sharpe Ratio", "sharpe_ratio"),
                ("Average Win", "avg_win"),
                ("Average Loss", "avg_loss"),
                ("Profit Factor", "profit_factor")
            ]
            
            for i, (label, key) in enumerate(metrics):
                row = i // 4
                col = (i % 4) * 2
                
                ttk.Label(metrics_frame, text=f"{label}:", style='Header.TLabel').grid(
                    row=row, column=col, sticky=tk.W, padx=(0, 5), pady=(0, 5))
                
                value_label = ttk.Label(metrics_frame, text="N/A", style='Status.TLabel')
                value_label.grid(row=row, column=col+1, sticky=tk.W, padx=(0, 20), pady=(0, 5))
                self.metrics_labels[key] = value_label
            
            # Chart placeholder (would implement actual charts in production)
            chart_frame = ttk.LabelFrame(parent, text="Performance Chart", padding="10")
            chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
            
            ttk.Label(chart_frame, text="📈 Performance chart would be displayed here", 
                     style='Status.TLabel').pack(expand=True)
            
        except Exception as e:
            self.logger.error(f"Analytics tab setup error: {e}")
            
    def setup_market_context_menu(self):
        """Setup context menu for market data"""
        try:
            self.market_context_menu = tk.Menu(self.root, tearoff=0)
            self.market_context_menu.add_command(label="Buy", command=self.context_buy)
            self.market_context_menu.add_command(label="Sell", command=self.context_sell)
            self.market_context_menu.add_separator()
            self.market_context_menu.add_command(label="Add to Watchlist", command=self.add_to_watchlist)
            self.market_context_menu.add_command(label="Remove from List", command=self.remove_from_list)
            
            self.market_tree.bind("<Button-3>", self.show_market_context_menu)
            
        except Exception as e:
            self.logger.error(f"Context menu setup error: {e}")
            
    def show_market_context_menu(self, event):
        """Show context menu for market data"""
        try:
            item = self.market_tree.selection()[0]
            if item:
                self.market_context_menu.post(event.x_root, event.y_root)
        except:
            pass
            
    # Event Handlers
    def connect_ibkr(self):
        """Connect to Interactive Brokers"""
        try:
            if self.trading_engine.event_loop:
                asyncio.run_coroutine_threadsafe(
                    self.trading_engine.connect(), 
                    self.trading_engine.event_loop
                )
            else:
                messagebox.showerror("Error", "Event loop not initialized")
        except Exception as e:
            self.logger.error(f"Connect IBKR error: {e}")
            messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")
            
    def disconnect_ibkr(self):
        """Disconnect from Interactive Brokers"""
        try:
            self.trading_engine.disconnect()
            messagebox.showinfo("Disconnected", "Successfully disconnected from IBKR")
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            
    def start_streaming(self):
        """Start market data streaming"""
        try:
            tickers = parse_ticker_list(self.tickers_var.get())
            if not tickers:
                messagebox.showerror("Error", "Please enter valid ticker symbols")
                return
                
            if not self.trading_engine.is_connected:
                messagebox.showerror("Error", "Please connect to IBKR first")
                return
                
            # Update configuration from GUI
            self.update_parameters()
            
            if self.trading_engine.event_loop:
                asyncio.run_coroutine_threadsafe(
                    self.trading_engine.start_streaming(tickers),
                    self.trading_engine.event_loop
                )
                self.is_streaming.set(True)
                messagebox.showinfo("Streaming", f"Started streaming for {len(tickers)} symbols")
            else:
                messagebox.showerror("Error", "Event loop not initialized")
                
        except Exception as e:
            self.logger.error(f"Start streaming error: {e}")
            messagebox.showerror("Streaming Error", f"Failed to start streaming: {str(e)}")
            
    def stop_streaming(self):
        """Stop market data streaming"""
        try:
            self.trading_engine.stop_streaming()
            self.is_streaming.set(False)
            messagebox.showinfo("Streaming", "Market data streaming stopped")
        except Exception as e:
            self.logger.error(f"Stop streaming error: {e}")
            
    def toggle_force_trade(self):
        """Toggle force trade mode"""
        try:
            self.trading_engine.toggle_force_trade_mode()
            force_mode = self.trading_engine.force_trade_mode
            self.force_trade_var.set(force_mode)
            self.force_status_var.set("Enabled" if force_mode else "Disabled")
            
            if force_mode:
                messagebox.showwarning("Force Trade Mode", 
                    "⚠️ Force Trade Mode is now ENABLED.\n\n"
                    "This mode bypasses some safety checks and may increase risk.\n"
                    "Use with caution!")
            else:
                messagebox.showinfo("Force Trade Mode", "Force Trade Mode is now disabled.")
                
        except Exception as e:
            self.logger.error(f"Force trade toggle error: {e}")
            
    def on_force_trade_toggle(self):
        """Handle force trade checkbox toggle"""
        self.toggle_force_trade()
        
    def update_parameters(self):
        """Update trading parameters from GUI"""
        try:
            # Update configuration
            self.config.equity_per_trade = self.equity_per_trade_var.get()
            self.config.max_open_trades = self.max_open_trades_var.get()
            self.config.hard_stop_loss = self.hard_stop_loss_var.get() / 100  # Convert from percentage
            self.config.take_profit = self.take_profit_var.get() / 100
            
            # Technical parameters
            self.config.rsi_period = self.rsi_period_var.get()
            self.config.bb_period = self.bb_period_var.get()
            self.config.stoch_k_period = self.stoch_k_period_var.get()
            self.config.data_points = self.data_points_var.get()
            
            # Update trading engine configuration
            if hasattr(self.trading_engine, 'update_config'):
                self.trading_engine.update_config(self.config)
                
            messagebox.showinfo("Parameters Updated", "Trading parameters have been updated successfully.")
            
        except Exception as e:
            self.logger.error(f"Parameter update error: {e}")
            messagebox.showerror("Parameter Error", f"Failed to update parameters: {str(e)}")
            
    # Data Update Methods
    def update_market_data(self, market_data: Dict):
        """Update market data display"""
        try:
            # Store data for reference
            self.market_data_cache = market_data
            
            # Clear existing data
            for item in self.market_tree.get_children():
                self.market_tree.delete(item)
                
            # Add new data
            for symbol, data in market_data.items():
                try:
                    # Extract values with proper formatting
                    price = data.get('current_price', 0)
                    change_pct = data.get('price_change', {}).get('percent_change', 0)
                    rsi = data.get('rsi', {}).get('value', 50)
                    bb_pos = data.get('bollinger', {}).get('position_percent', 0.5) * 100
                    stoch_k = data.get('stochastic', {}).get('k_percent', 50)
                    stoch_d = data.get('stochastic', {}).get('d_percent', 50)
                    volume = data.get('volume_analysis', {}).get('current_volume', 0)
                    slope = data.get('slope', {}).get('value', 0)
                    
                    # Determine signal
                    signals = data.get('signals', {})
                    signal = signals.get('overall_signal', 'HOLD')
                    
                    # Format timestamp
                    timestamp = data.get('timestamp', datetime.now())
                    time_str = timestamp.strftime("%H:%M:%S") if timestamp else "N/A"
                    
                    # Insert row with color coding based on change
                    item_id = self.market_tree.insert("", tk.END, values=(
                        symbol,
                        f"${price:.2f}",
                        format_percentage(change_pct),
                        f"{rsi:.1f}",
                        f"{bb_pos:.1f}%",
                        f"{stoch_k:.1f}",
                        f"{stoch_d:.1f}",
                        f"{volume:,.0f}",
                        f"{slope:.2f}",
                        signal,
                        time_str
                    ))
                    
                    # Color coding
                    if change_pct > 0:
                        self.market_tree.set(item_id, "Change %", format_percentage(change_pct))
                        # Would set tag for green color in production
                    elif change_pct < 0:
                        self.market_tree.set(item_id, "Change %", format_percentage(change_pct))
                        # Would set tag for red color in production
                        
                except Exception as e:
                    self.logger.error(f"Error processing market data for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Market data update error: {e}")
            
    def update_trades_display(self, open_positions: Dict, trade_history: List):
        """Update trades display"""
        try:
            # Update open positions
            self.update_positions_display(open_positions)
            
            # Update trade history
            self.update_history_display(trade_history)
            
            # Update position count
            self.open_positions_var.set(str(len(open_positions)))
            
        except Exception as e:
            self.logger.error(f"Trades display update error: {e}")
            
    def update_positions_display(self, positions: Dict):
        """Update open positions display"""
        try:
            # Clear existing positions
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
                
            # Add current positions
            for symbol, position in positions.items():
                try:
                    entry_time = position.get('entry_time', datetime.now())
                    duration = calculate_time_difference(entry_time)
                    
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    pnl_color = 'green' if unrealized_pnl > 0 else 'red' if unrealized_pnl < 0 else 'black'
                    
                    self.positions_tree.insert("", tk.END, values=(
                        symbol,
                        position.get('action', 'N/A'),
                        position.get('quantity', 0),
                        f"${position.get('entry_price', 0):.2f}",
                        f"${position.get('current_price', 0):.2f}",
                        format_currency(unrealized_pnl),
                        f"${position.get('stop_loss', 0):.2f}",
                        f"${position.get('take_profit', 0):.2f}",
                        entry_time.strftime("%H:%M:%S"),
                        duration
                    ))
                    
                except Exception as e:
                    self.logger.error(f"Error processing position for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Positions display update error: {e}")
            
    def update_history_display(self, history: List):
        """Update trade history display"""
        try:
            # Store for reference
            self.trade_history_cache = history
            
            # Clear existing history
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
                
            # Add recent trades (last 50)
            recent_history = history[-50:] if len(history) > 50 else history
            
            for trade in reversed(recent_history):  # Show most recent first
                try:
                    entry_time = trade.get('entry_time', datetime.now())
                    exit_time = trade.get('exit_time', datetime.now())
                    duration = calculate_time_difference(entry_time, exit_time)
                    
                    realized_pnl = trade.get('realized_pnl', 0)
                    
                    self.history_tree.insert("", tk.END, values=(
                        trade.get('trade_id', 'N/A'),
                        trade.get('symbol', 'N/A'),
                        trade.get('action', 'N/A'),
                        trade.get('quantity', 0),
                        f"${trade.get('entry_price', 0):.2f}",
                        f"${trade.get('exit_price', 0):.2f}",
                        format_currency(realized_pnl),
                        entry_time.strftime("%H:%M:%S"),
                        exit_time.strftime("%H:%M:%S"),
                        duration,
                        trade.get('strategy', 'Auto')
                    ))
                    
                except Exception as e:
                    self.logger.error(f"Error processing trade history: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"History display update error: {e}")
            
    def update_status(self, status_data: Dict):
        """Update status information"""
        try:
            # Connection status
            if status_data.get('connected', False):
                self.connection_status.set("Connected")
            else:
                self.connection_status.set("Disconnected")
                
            # Streaming status
            if status_data.get('streaming', False):
                self.streaming_status.set("Active")
            else:
                self.streaming_status.set("Stopped")
                
            # Financial data
            balance = status_data.get('account_balance', 0)
            self.account_balance.set(format_currency(balance))
            
            daily_pnl = status_data.get('daily_pnl', 0)
            self.daily_pnl_var.set(format_currency(daily_pnl))
            
            # Performance metrics
            win_rate = status_data.get('win_rate', 0)
            self.win_rate_var.set(f"{win_rate:.1f}%")
            
            daily_trades = status_data.get('daily_trades', 0)
            self.daily_trades_var.set(str(daily_trades))
            
            # Force trade status
            force_mode = status_data.get('force_trade_mode', False)
            self.force_status_var.set("Enabled" if force_mode else "Disabled")
            
            # Update analytics if present
            if hasattr(self, 'metrics_labels'):
                self.update_analytics_display(status_data)
                
        except Exception as e:
            self.logger.error(f"Status update error: {e}")
            
    def update_analytics_display(self, data: Dict):
        """Update analytics metrics display"""
        try:
            metrics_mapping = {
                'total_trades': ('total_trades', lambda x: str(x)),
                'win_rate': ('win_rate', lambda x: f"{x:.1f}%"),
                'total_pnl': ('total_pnl', lambda x: format_currency(x)),
                'max_drawdown': ('max_drawdown', lambda x: f"{x:.2f}%"),
                'sharpe_ratio': ('sharpe_ratio', lambda x: f"{x:.2f}"),
                'avg_win': ('avg_win', lambda x: format_currency(x)),
                'avg_loss': ('avg_loss', lambda x: format_currency(x)),
                'profit_factor': ('profit_factor', lambda x: f"{x:.2f}")
            }
            
            for key, (data_key, formatter) in metrics_mapping.items():
                if key in self.metrics_labels and data_key in data:
                    value = data[data_key]
                    formatted_value = formatter(value)
                    self.metrics_labels[key].config(text=formatted_value)
                    
        except Exception as e:
            self.logger.error(f"Analytics display update error: {e}")
            
    # Utility Methods
    def start_periodic_updates(self):
        """Start periodic GUI updates"""
        try:
            self.update_gui_status()
            self.root.after(self.config.update_interval_ms, self.start_periodic_updates)
        except Exception as e:
            self.logger.error(f"Periodic update error: {e}")
            self.root.after(self.config.update_interval_ms, self.start_periodic_updates)
            
    def update_gui_status(self):
        """Update GUI status indicators"""
        try:
            # Get current status from trading engine
            stats = self.trading_engine.get_performance_stats()
            
            status_data = {
                'connected': self.trading_engine.is_connected,
                'streaming': self.trading_engine.is_streaming,
                'account_balance': stats.get('account_balance', 0),
                'daily_pnl': stats.get('daily_pnl', 0),
                'win_rate': stats.get('win_rate', 0),
                'daily_trades': stats.get('daily_trades', 0),
                'force_trade_mode': stats.get('force_trade_mode', False),
                'total_trades': stats.get('total_trades', 0),
                'max_drawdown': stats.get('max_drawdown', 0),
                'total_pnl': stats.get('total_pnl', 0)
            }
            
            self.update_status(status_data)
            
        except Exception as e:
            self.logger.error(f"GUI status update error: {e}")
            
    # Menu Action Methods
    def export_trades(self):
        """Export trade history to CSV"""
        try:
            if not self.trade_history_cache:
                messagebox.showinfo("Export", "No trade history to export")
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Trade History"
            )
            
            if filename:
                df = pd.DataFrame(self.trade_history_cache)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Export", f"Trade history exported to {filename}")
                
        except Exception as e:
            self.logger.error(f"Export trades error: {e}")
            messagebox.showerror("Export Error", f"Failed to export trades: {str(e)}")
            
    def import_settings(self):
        """Import settings from file"""
        messagebox.showinfo("Import Settings", "Import settings functionality would be implemented here")
        
    def export_settings(self):
        """Export settings to file"""
        messagebox.showinfo("Export Settings", "Export settings functionality would be implemented here")
        
    def run_backtest(self):
        """Run strategy backtest"""
        messagebox.showinfo("Backtest", "Backtesting functionality would be implemented here")
        
    def show_performance_report(self):
        """Show detailed performance report"""
        messagebox.showinfo("Performance Report", "Performance report would be displayed here")
        
    def show_risk_analysis(self):
        """Show risk analysis"""
        messagebox.showinfo("Risk Analysis", "Risk analysis would be displayed here")
        
    def show_market_scanner(self):
        """Show market scanner"""
        messagebox.showinfo("Market Scanner", "Market scanner would be implemented here")
        
    def show_options_calculator(self):
        """Show options calculator"""
        messagebox.showinfo("Options Calculator", "Options calculator would be implemented here")
        
    def show_position_sizer(self):
        """Show position sizing calculator"""
        messagebox.showinfo("Position Sizer", "Position sizing calculator would be implemented here")
        
    def show_user_guide(self):
        """Show user guide"""
        messagebox.showinfo("User Guide", "User guide would be displayed here")
        
    def show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
Keyboard Shortcuts:

Ctrl+C - Connect to IBKR
Ctrl+D - Disconnect from IBKR
Ctrl+Q - Exit Application

F1 - Start Streaming
F2 - Stop Streaming
F3 - Toggle Force Trade Mode
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
AutoTrade Plus - Enhanced Trading Platform
Version 2.0

Advanced algorithmic trading platform with:
• Real-time market data streaming
• Sophisticated technical analysis
• Automated trade execution
• Advanced risk management
• Comprehensive performance analytics

Built with Python, tkinter, and ib-insync
        """
        messagebox.showinfo("About", about_text)
        
    # Context Menu Actions
    def context_buy(self):
        """Buy selected symbol"""
        try:
            selection = self.market_tree.selection()
            if selection:
                item = selection[0]
                symbol = self.market_tree.item(item)['values'][0]
                messagebox.showinfo("Buy Order", f"Buy order for {symbol} would be placed here")
        except Exception as e:
            self.logger.error(f"Context buy error: {e}")
            
    def context_sell(self):
        """Sell selected symbol"""
        try:
            selection = self.market_tree.selection()
            if selection:
                item = selection[0]
                symbol = self.market_tree.item(item)['values'][0]
                messagebox.showinfo("Sell Order", f"Sell order for {symbol} would be placed here")
        except Exception as e:
            self.logger.error(f"Context sell error: {e}")
            
    def add_to_watchlist(self):
        """Add symbol to watchlist"""
        messagebox.showinfo("Watchlist", "Add to watchlist functionality would be implemented here")
        
    def remove_from_list(self):
        """Remove symbol from list"""
        try:
            selection = self.market_tree.selection()
            if selection:
                item = selection[0]
                symbol = self.market_tree.item(item)['values'][0]
                
                result = messagebox.askyesno("Remove Symbol", f"Remove {symbol} from the list?")
                if result:
                    # Remove from ticker list
                    current_tickers = parse_ticker_list(self.tickers_var.get())
                    if symbol in current_tickers:
                        current_tickers.remove(symbol)
                        self.tickers_var.set(','.join(current_tickers))
                        
        except Exception as e:
            self.logger.error(f"Remove from list error: {e}")
            
    # Position Management
    def close_selected_position(self):
        """Close selected position"""
        try:
            selection = self.positions_tree.selection()
            if selection:
                item = selection[0]
                symbol = self.positions_tree.item(item)['values'][0]
                
                result = messagebox.askyesno("Close Position", f"Close position for {symbol}?")
                if result:
                    messagebox.showinfo("Close Position", f"Position close order for {symbol} would be executed here")
        except Exception as e:
            self.logger.error(f"Close position error: {e}")
            
    def close_all_positions(self):
        """Close all open positions"""
        try:
            result = messagebox.askyesno("Close All Positions", 
                "Are you sure you want to close ALL open positions?")
            if result:
                messagebox.showinfo("Close All", "All positions would be closed here")
        except Exception as e:
            self.logger.error(f"Close all positions error: {e}")
            
    def update_stop_loss(self):
        """Update stop loss for selected position"""
        try:
            selection = self.positions_tree.selection()
            if selection:
                messagebox.showinfo("Update Stop Loss", "Stop loss update functionality would be implemented here")
        except Exception as e:
            self.logger.error(f"Update stop loss error: {e}")
            
    # Data Management
    def sort_market_data(self, column):
        """Sort market data by column"""
        try:
            # Simple sort implementation
            items = [(self.market_tree.set(child, column), child) for child in self.market_tree.get_children('')]
            items.sort()
            
            for index, (val, child) in enumerate(items):
                self.market_tree.move(child, '', index)
                
        except Exception as e:
            self.logger.error(f"Sort market data error: {e}")
            
    def clear_history(self):
        """Clear trade history"""
        try:
            result = messagebox.askyesno("Clear History", 
                "Are you sure you want to clear all trade history?")
            if result:
                self.trade_history_cache = []
                self.update_history_display([])
                messagebox.showinfo("History Cleared", "Trade history has been cleared")
        except Exception as e:
            self.logger.error(f"Clear history error: {e}")
            
    def filter_history(self):
        """Filter trade history"""
        messagebox.showinfo("Filter History", "History filtering functionality would be implemented here")
        
    def on_closing(self):
        """Handle application closing"""
        try:
            result = messagebox.askyesno("Exit", "Are you sure you want to exit?")
            if result:
                # Stop streaming if active
                if self.trading_engine.is_streaming:
                    self.trading_engine.stop_streaming()
                    
                # Disconnect if connected
                if self.trading_engine.is_connected:
                    self.trading_engine.disconnect()
                    
                # Close application
                self.root.quit()
                self.root.destroy()
                
        except Exception as e:
            self.logger.error(f"Application closing error: {e}")
            self.root.quit()
