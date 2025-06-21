from ib_insync import IB, Stock, Order
import tkinter as tk
from tkinter import ttk
import pandas as pd
import threading
import time
import asyncio
import nest_asyncio
import os
from datetime import datetime
from pytz import timezone
import pytz
from PIL import Image, ImageTk
import os
import numpy as np
from sklearn.linear_model import LinearRegression
nest_asyncio.apply()  # Allow nested async event loops


def compute_slope(prices):
    x = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0][0]

def calculate_adaptive_thresholds(spike_stats):
    """
    Calculate adaptive thresholds based on collected spike metrics.
    Only works if at least 3 spikes have been recorded.
    """
    if spike_stats["count"] < 3:
        return None

    return {
        "price_rate": sum(spike_stats["price_rates"]) / len(spike_stats["price_rates"]),
        "rsi_rate": sum(spike_stats["rsi_rates"]) / len(spike_stats["rsi_rates"]),
        "boll_rate": sum(spike_stats["boll_rates"]) / len(spike_stats["boll_rates"]),
        "slope": sum(spike_stats["slopes"]) / len(spike_stats["slopes"])
    }

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Trading Analysis")
        self.root.geometry("1200x600")

        # Initial flags and connection
        self.trade_intent_log = []
        self.is_streaming = False
        self.ib = IB()
        self.nextOrderId = 1
        self.trade_cooldowns = {}
        self.max_open_trades = 2
        self.force_trade = tk.BooleanVar(value=False)

        # User Inputs
        self.tickers = tk.StringVar(value="ABTS")
        self.data_points = tk.IntVar(value=15)
        self.chart_interval = tk.StringVar(value="5 mins")
        self.analysis_interval = tk.IntVar(value=1)
        self.rsi_period = tk.IntVar(value=2)
        self.bb_period = tk.IntVar(value=2)
        self.stoch_k_period = tk.IntVar(value=2)
        self.stoch_d_period = tk.IntVar(value=2)
        self.stoch_k_min = tk.DoubleVar(value=50)
        self.stoch_d_min = tk.DoubleVar(value=40)
        self.duration_str = tk.StringVar(value="1 D")
        self.equity_per_trade = tk.DoubleVar(value=100.0)
        self.exchange_var = tk.StringVar(value="SMART")
        self.available_funds = self.equity_per_trade.get()
        self.hard_stop_loss = tk.DoubleVar(value=0.03)  # 👈 Add this line here
        self.last_price_change = "N/A"
        self.last_rsi_change = "N/A"
        self.last_bollinger_change = "N/A"
        self.ib.orderStatusEvent += self.on_order_status
        # Data and orders
        self.market_data = {}
        self.active_orders = {}

        # Setup UI and connect
        self.setup_gui()
        self.connect_ibkr()
        self.update_trades_ui()  # ✅ Add this line

    def compute_rsi(self, df, period=14):
        """ Compute the Relative Strength Index (RSI) based on historical price data. """
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_stochastic(self, df, k_period=14, d_period=3):
        low_min = df['close'].rolling(window=k_period).min()
        high_max = df['close'].rolling(window=k_period).max()
        df['%K'] = (df['close'] - low_min) / (high_max - low_min) * 100
        df['%D'] = df['%K'].rolling(window=d_period).mean()
        return df


    def compute_bollinger(self, df, period=20, num_std=2):
        """ Compute Bollinger % Position based on historical price data. """
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bollinger_pct = (df["close"] - lower_band) / (upper_band - lower_band) * 100
        return bollinger_pct


    def interval_to_minutes(self):
        """Converts selected chart interval string to number of minutes."""
        mapping = {
            "5 sec": 5 / 60,
            "10 sec": 10 / 60,
            "30 sec": 30 / 60,
            "1 min": 1,
            "3 min": 3,
            "5 min": 5,
            "10 min": 10,
            "15 min": 15
        }
        return mapping.get(self.chart_interval.get(), 1)

    def connect_ibkr(self):
        """ Connect to IBKR API with predefined login details """
        try:
            username = "hsaidx343"
            password = "uasfu29y8489(G(G("
            self.ib.connect("127.0.0.1", 7497, clientId=2)
            print(f"✅ Connected to IBKR API as user {username}")
        except Exception as e:
            print(f"❌ IBKR Connection Error: {e}")
            return

    def setup_gui(self):
        """ Updated UI Layout with Banner and working output widgets """
        self.root.configure(bg="#FFFFFF")
        self.root.geometry("800x600")

        # --- BANNER ---
        try:
            banner_img_path = os.path.join("C:/app/assets/images", "autotrade_plus_banner.png")
            banner_image = Image.open(banner_img_path)
            banner_image = banner_image.resize((600, 180), Image.Resampling.LANCZOS)
            banner_photo = ImageTk.PhotoImage(banner_image)
            banner_label = tk.Label(self.root, image=banner_photo, bg="#FFFFFF")
            banner_label.image = banner_photo
            banner_label.pack(pady=(0, 0))
        except Exception as e:
            print(f"❌ Failed to load banner image: {e}")
            banner_label = tk.Label(self.root, text="Autotrade Plus", font=("Nunito", 24, "bold"), bg="#FFFFFF")
            banner_label.pack(pady=(0, 0))

        # --- MAIN CONTAINER ---
        container = tk.Frame(self.root, bg="#FFFFFF")
        container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # --- INPUTS: Two columns ---
        input_frame = tk.Frame(container, bg="#FFFFFF")
        input_frame.pack(pady=0, fill=tk.X)

        left = tk.Frame(input_frame, bg="#ffffff")
        right = tk.Frame(input_frame, bg="#ffffff")
        left.pack(side=tk.LEFT, padx=0, expand=True)
        right.pack(side=tk.RIGHT, padx=0, expand=True)

        inputs = [
            ("Tickers:", self.tickers, 30),
            ("Exchange:", self.exchange_var, ["SMART", "NYSE", "NASDAQ"]),
            ("Chart Interval:", self.chart_interval, ["1 min", "3 min", "5 min", "15 min"]),
            ("Analysis Interval:", self.analysis_interval, 5),
            ("RSI Period:", self.rsi_period, 5),
            ("Bollinger Period:", self.bb_period, 5),
            ("Stoch %K Period:", self.stoch_k_period, 5),
            ("Stoch %D Period:", self.stoch_d_period, 5),
            ("Stoch %K Min:", self.stoch_k_min, 5),
            ("Stoch %D Min:", self.stoch_d_min, 5),
            ("Duration:", self.duration_str, 10),
            ("Equity Per Trade:", self.equity_per_trade, 10)
        ]

        half = len(inputs) // 2
        for i, (label, var, conf) in enumerate(inputs[:half]):
            tk.Label(left, text=label, bg="#ffffff").grid(row=i, column=0, sticky="w", pady=0)
            if isinstance(conf, list):
                ttk.Combobox(left, textvariable=var, values=conf).grid(row=i, column=1, pady=0)
            else:
                tk.Entry(left, textvariable=var, width=conf).grid(row=i, column=1, pady=0)

        for i, (label, var, conf) in enumerate(inputs[half:]):
            tk.Label(right, text=label, bg="#ffffff").grid(row=i, column=0, sticky="w", pady=0)
            if isinstance(conf, list):
                ttk.Combobox(right, textvariable=var, values=conf).grid(row=i, column=1, pady=0)
            else:
                tk.Entry(right, textvariable=var, width=conf).grid(row=i, column=1, pady=0)

        # --- BUTTONS ---
        button_frame = tk.Frame(container, bg="#ffffff")
        button_frame.pack(pady=0)
        actions = [
            ("Start Streaming", self.start_stream),
            ("Stop Streaming", self.stop_stream),
            ("Force Trade", self.toggle_force_trade),
            ("Run Backtest", self.run_backtest)
        ]
        for text, cmd in actions:
            tk.Button(button_frame, text=text, command=cmd, bg="#007BFF", fg="white",
                    font=("Nunito", 10, "bold"), width=18).pack(side=tk.LEFT, padx=0)

        # --- MARKET DATA OUTPUT ---
        data_frame = tk.LabelFrame(self.root, text="Live Market Feed", bg="#ffffff", font=("Nunito", 12, "bold"))
        data_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.tree = ttk.Treeview(data_frame, columns=(
            "Symbol", "Exchange", "Last Price", "ΔP", "ΔRSI", "ΔBB", "%K", "%D", "Status"
        ), show="headings")

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

        # --- TRADE SUMMARY OUTPUT ---
        trades_frame = tk.LabelFrame(self.root, text="Open Trades", bg="#ffffff", font=("Nunito", 12, "bold"))
        trades_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.trades_tree = ttk.Treeview(trades_frame, columns=(
            "Symbol", "Quantity", "Average Price", "Current Price", "Unrealized PnL"
        ), show="headings")
        for col in self.trades_tree["columns"]:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, anchor="center")
        self.trades_tree.pack(fill=tk.BOTH, expand=True)


    def update_trades_ui(self):
        """ Update the trades panel with live open positions """
        self.trades_tree.delete(*self.trades_tree.get_children())  # Clear existing data

        try:
            positions = []
            for symbol, trade in self.active_orders.items():
                if not trade.get("sold", False):
                    positions.append({
                        "symbol": symbol,
                        "quantity": trade["quantity"],
                        "avg_price": trade["buy_price"],
                        "current_price": self.market_data.get(symbol, {}).get('last')
                    })
            
            for pos in positions:
                symbol = pos["symbol"]
                quantity = pos["quantity"]
                avg_price = pos["avg_price"]
                current_price = pos["current_price"] or 0.0
                unrealized_pnl = (current_price - avg_price) * quantity

                self.trades_tree.insert("", "end", values=(
                    symbol,
                    quantity,
                    f"${avg_price:.2f}",
                    f"${current_price:.2f}",
                    f"${unrealized_pnl:.2f}"
                ))

   
            market_data = self.market_data  # Shortcut

            for pos in positions:
                symbol = pos.contract.symbol
                quantity = pos.position
                avg_price = pos.avgCost

                # Try to get the latest price from market data
                current_price = market_data.get(symbol, {}).get('last', None)

                if current_price is not None:
                    unrealized_pnl = (current_price - avg_price) * quantity
                else:
                    unrealized_pnl = 0.0

                self.trades_tree.insert("", "end", values=(
                    symbol,
                    quantity,
                    f"${avg_price:.2f}",
                    f"${current_price:.2f}" if current_price is not None else "N/A",
                    f"${unrealized_pnl:.2f}"
                ))

        except Exception as e:
            print(f"❌ Error updating trades UI: {e}")

        # Schedule the next update after 5 seconds
        self.root.after(5000, self.update_trades_ui)

                    
    def on_order_status(self, trade):
        """
        Callback for IBKR order status updates.
        Automatically marks trades as sold and updates funds + cooldown timer.
        """
        print(f"📬 Order update: {trade.orderStatus.status} | Symbol: {trade.contract.symbol}")

        if trade.orderStatus.status == 'Filled':
            symbol = trade.contract.symbol

            if symbol in self.active_orders and not self.active_orders[symbol].get('sold', False):
                quantity = self.active_orders[symbol]['quantity']
                price = trade.order.lmtPrice if trade.order.lmtPrice else 0
                recovered_funds = quantity * price

                self.active_orders[symbol]['sold'] = True
                self.available_funds += recovered_funds
                self.trade_cooldowns[symbol] = datetime.now()

                print(f"✅ {symbol} filled at ${price:.2f}. Recovered ${recovered_funds:.2f}.")
                print(f"⏳ Cooldown started for {symbol}.")


    def run_backtest(self):
        import os
        from datetime import datetime

        tickers = self.tickers.get().split(",")
        exchange = self.exchange_var.get()
        duration_str = self.duration_str.get()
        equity_per_trade = self.equity_per_trade.get()
        analysis_interval = self.analysis_interval.get()

        total_trades = 0
        total_profit = 0
        total_equity_used = 0
        initial_funds = 1000
        available_funds = initial_funds
        trade_log = []

        print("📊 Running backtest...")

        for symbol in tickers:
            try:
                contract = Stock(symbol.strip(), exchange, "USD")
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=duration_str,
                    barSizeSetting=self.chart_interval.get(),
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1
                )

                df = pd.DataFrame([{"date": bar.date, "close": bar.close} for bar in bars])
                df["RSI"] = self.compute_rsi(df)
                df["Bollinger %"] = self.compute_bollinger(df)
                df = self.compute_stochastic(df, k_period=self.stoch_k_period.get(), d_period=self.stoch_d_period.get())

                trend_window = 5
                df["trend_slope"] = df["close"].rolling(trend_window).apply(compute_slope, raw=False)

                if len(df) < 21:
                    continue

                i = analysis_interval
                while i < len(df) - 1:
                    window = self.analysis_interval.get()
                    minutes_per_bar = self.interval_to_minutes()

                    df["price_pct"] = df["close"].pct_change(fill_method=None) * 100
                    df["price_rate"] = (df["price_pct"] / minutes_per_bar).rolling(window=window).mean()
                    df["rsi_pct"] = df["RSI"].pct_change(fill_method=None) * 100
                    df["rsi_rate"] = (df["rsi_pct"] / minutes_per_bar).rolling(window=window).mean()
                    df["bollinger_pct"] = df["Bollinger %"].pct_change(fill_method=None) * 100
                    df["bollinger_rate"] = (df["bollinger_pct"] / minutes_per_bar).rolling(window=window).mean()

                    price_change = df["price_rate"].iloc[i]
                    rsi_change = df["rsi_rate"].iloc[i]
                    bollinger_change = df["bollinger_rate"].iloc[i]

                    # ✅ Skip if trend is weak
                    if df["trend_slope"].iloc[i] < 0.02:
                        i += 1
                        continue

                    if (
                        price_change >= 0.257142857 and
                        rsi_change >= 0.201904762 and
                        bollinger_change >= 1.48571429 and
                        df["%K"].iloc[i] > df["%D"].iloc[i] and
                        df["%K"].iloc[i - 1] <= df["%D"].iloc[i - 1] and
                        df["%K"].iloc[i] >= self.stoch_k_min.get() and
                        df["%D"].iloc[i] >= self.stoch_d_min.get()
                    ):
                        entry_price = df.iloc[i]["close"]
                        trade_cost = equity_per_trade

                        if available_funds >= trade_cost:
                            shares = int(trade_cost // entry_price)
                            used_capital = shares * entry_price
                            available_funds -= used_capital

                            max_price = entry_price
                            profit_floor = entry_price
                            exit_price = entry_price
                            reason_for_exit = "Hold to End"
                            max_drawdown = 0.02
                            profit_step = 0.02
                            max_hold_seconds = 180
                            bar_interval_seconds = self.interval_to_minutes() * 60
                            elapsed_time = 0

                            j = i + 1
                            while j < len(df) and elapsed_time <= max_hold_seconds:
                                future_price = df.iloc[j]["close"]

                                if future_price > max_price:
                                    max_price = future_price
                                    profit_pct = (max_price - entry_price) / entry_price
                                    step_count = int(profit_pct / profit_step)
                                    new_floor = round(entry_price * (1 + step_count * profit_step), 2)
                                    if new_floor > profit_floor:
                                        profit_floor = new_floor

                                if (future_price - entry_price) / entry_price <= -0.03:
                                    exit_price = future_price
                                    reason_for_exit = "Hard Stop-Loss"
                                    break

                                if (future_price - max_price) / max_price <= -max_drawdown:
                                    exit_price = future_price
                                    reason_for_exit = "Trailing Stop"
                                    break

                                if future_price < profit_floor:
                                    exit_price = future_price
                                    reason_for_exit = "Profit Floor"
                                    break

                                j += 1
                                elapsed_time += bar_interval_seconds

                            if j == len(df) or elapsed_time > max_hold_seconds:
                                exit_price = df.iloc[j - 1]["close"]
                                reason_for_exit = "Time Limit"

                            profit = (exit_price - entry_price) * shares
                            available_funds += used_capital + profit
                            total_trades += 1
                            total_profit += profit
                            total_equity_used += used_capital

                            buy_time = df.iloc[i]["date"]
                            sell_time = df.iloc[j if j < len(df) else -1]["date"]
                            profit_pct = ((exit_price - entry_price) / entry_price) * 100

                            trade_log.append({
                                "Symbol": symbol.strip(),
                                "Buy Price": round(entry_price, 2),
                                "Sell Price": round(exit_price, 2),
                                "Buy Date/Time": buy_time,
                                "Sell Date/Time": sell_time,
                                "Profit ($)": round(profit, 2),
                                "Profit (%)": round(profit_pct, 2),
                                "Exit Reason": reason_for_exit,
                                "Price Rate": round(price_change, 4),
                                "RSI Rate": round(rsi_change, 4),
                                "Bollinger Rate": round(bollinger_change, 4),
                                "Trend Slope": round(df["trend_slope"].iloc[i], 5)
                            })
        
                    i += 1

            except Exception as e:
                print(f"❌ Error during backtest for {symbol.strip()}: {e}")

        self.show_backtest_popup(trade_log, total_trades, total_profit, total_equity_used)

    def show_backtest_popup(self, trade_log, total_trades, total_profit, total_equity, adaptive_info=None):
        popup = tk.Toplevel(self.root)
        popup.title("Backtest Results")
        popup.geometry("1000x700")
        popup.configure(bg="#ffffff")

        text_widget = tk.Text(popup, wrap="none", font=("Courier New", 10), bg="#f8f8f8", height=20)
        text_widget.pack(expand=True, fill="both", padx=10, pady=10)

        # === Adaptive notice ===
        if adaptive_info and adaptive_info.get("switched_at"):
            text_widget.insert("end", f"⚙️ Adaptive Mode Activated At: {adaptive_info['switched_at']}\n")
            thresholds = adaptive_info.get("thresholds", {})
            if thresholds:
                text_widget.insert("end", f"🧠 New Thresholds → "
                                        f"Price: {thresholds['price_rate']:.3f}, "
                                        f"RSI: {thresholds['rsi_rate']:.3f}, "
                                        f"Boll: {thresholds['boll_rate']:.3f}, "
                                        f"Slope: {thresholds['slope']:.5f}\n\n")

                # === Adaptive threshold totals ===
                text_widget.insert("end", f"📊 Total Used in Threshold Calc:\n")
                text_widget.insert("end", f"   • Price Rate Avg: {thresholds['price_rate']:.4f}\n")
                text_widget.insert("end", f"   • RSI Rate Avg  : {thresholds['rsi_rate']:.4f}\n")
                text_widget.insert("end", f"   • Boll Rate Avg : {thresholds['boll_rate']:.4f}\n")
                text_widget.insert("end", f"   • Slope Avg     : {thresholds['slope']:.5f}\n\n")
        else:
            text_widget.insert("end", f"🛑 Adaptive Mode Never Triggered\n\n")

        header = (
            f"📊 Backtest Summary\n"
            f"🧾 Total Trades      : {total_trades}\n"
            f"💰 Total Profit      : ${total_profit:.2f}\n"
            f"📦 Total Equity Used : ${total_equity:.2f}\n\n"
            f"{'Symbol':<8} {'Buy Time':<19} {'Sell Time':<19} {'Buy':>6} {'Sell':>6} {'PnL $':>8} {'PnL %':>7} {'Reason':<14}\n"
            + "-" * 100 + "\n"
        )
        text_widget.insert("end", header)

        for t in trade_log:
            row = (
                f"{t['Symbol']:<8} {str(t['Buy Date/Time'])[:19]:<19} {str(t['Sell Date/Time'])[:19]:<19} "
                f"${t['Buy Price']:>5.2f} ${t['Sell Price']:>5.2f} "
                f"${t['Profit ($)']:>7.2f} {t['Profit (%)']:>6.2f}% {t['Exit Reason']:<14} [{t['Mode']}]\n"
            )
            diagnostics = (
                f"   ↳ Rates → Price: {t['Price Rate']:.4f} | RSI: {t['RSI Rate']:.4f} | "
                f"Bollinger: {t['Bollinger Rate']:.4f} | Slope: {t['Trend Slope']:.5f}\n"
            )

            text_widget.insert("end", row)
            text_widget.insert("end", diagnostics)

        text_widget.config(state="disabled")

        # === Spike Viewer Dropdown ===
        if adaptive_info and "spikes" in adaptive_info:
            spike_data = adaptive_info["spikes"]  # list of (price, rsi, boll, slope)

            def show_spike_details(event=None):
                selected_index = spike_combo.current()
                if selected_index >= 0:
                    p, r, b, s = spike_data[selected_index]
                    spike_text.config(state="normal")
                    spike_text.delete("1.0", "end")
                    spike_text.insert("end", f"📍 Spike #{selected_index + 1} Details\n")
                    spike_text.insert("end", f"   • Price Rate : {p:.4f}\n")
                    spike_text.insert("end", f"   • RSI Rate   : {r:.4f}\n")
                    spike_text.insert("end", f"   • Boll Rate  : {b:.4f}\n")
                    spike_text.insert("end", f"   • Slope      : {s:.5f}\n")
                    spike_text.config(state="disabled")

            spike_frame = tk.Frame(popup, bg="#ffffff")
            spike_frame.pack(pady=(10, 0))

            tk.Label(spike_frame, text="🧪 Select Spike to Inspect:", bg="#ffffff").pack(side=tk.LEFT)
            spike_combo = ttk.Combobox(spike_frame, values=[f"Spike #{i+1}" for i in range(len(spike_data))])
            spike_combo.pack(side=tk.LEFT, padx=10)
            spike_combo.bind("<<ComboboxSelected>>", show_spike_details)

            spike_text = tk.Text(popup, height=6, width=60, font=("Courier New", 10), bg="#f0f0f0")
            spike_text.pack(pady=(5, 10))
            spike_text.config(state="disabled")




    def update_ui(self):
        """ Update UI with real-time data and apply backtest logic for live decisions """
        self.tree.delete(*self.tree.get_children())  # Clear existing data
        self.trades_tree.delete(*self.trades_tree.get_children())  # Clear trade data

        for symbol, data in self.market_data.items():
            if "last" not in data or data["last"] is None:
                continue

            try:
                contract = Stock(symbol, 'SMART', 'USD')
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="1 D",
                    barSizeSetting=self.chart_interval.get(),
                    whatToShow="TRADES", useRTH=False, formatDate=1
                )

                if not bars or len(bars) < 20:
                    continue

                df = pd.DataFrame([{"date": bar.date, "close": bar.close} for bar in bars])
                df["RSI"] = self.compute_rsi(df)
                df["Bollinger %"] = self.compute_bollinger(df)
                df = self.compute_stochastic(df, self.stoch_k_period.get(), self.stoch_d_period.get())

                window = self.analysis_interval.get()
                minutes_per_bar = self.interval_to_minutes()

                df["price_pct"] = df["close"].pct_change(fill_method=None) * 100
                df["price_rate"] = (df["price_pct"] / minutes_per_bar).rolling(window=window).mean()
                df["rsi_pct"] = df["RSI"].pct_change(fill_method=None) * 100
                df["rsi_rate"] = (df["rsi_pct"] / minutes_per_bar).rolling(window=window).mean()
                df["bollinger_pct"] = df["Bollinger %"].pct_change(fill_method=None) * 100
                df["bollinger_rate"] = (df["bollinger_pct"] / minutes_per_bar).rolling(window=window).mean()

                price_change = df["price_rate"].iloc[-1]
                rsi_change = df["rsi_rate"].iloc[-1]
                bollinger_change = df["bollinger_rate"].iloc[-1]
                current = df.iloc[-1]

                meets_criteria = (
                    price_change >= 0.257142857 and
                    rsi_change >= 0.201904762 and
                    bollinger_change >= 1.48571429 and
                    df["%K"].iloc[-1] > df["%D"].iloc[-1] and
                    df["%K"].iloc[-2] <= df["%D"].iloc[-2] and
                    df["%K"].iloc[-1] >= self.stoch_k_min.get() and
                    df["%D"].iloc[-1] >= self.stoch_d_min.get()
                )

                should_force_buy = self.force_trade.get()
                status = f"{symbol} meets criteria" if meets_criteria else f"{symbol} does not meet criteria"
                self.force_trade = tk.BooleanVar(value=False)

                if meets_criteria or should_force_buy:
                    now = datetime.now()
                    cooldown_period = 300
                    last_trade_time = self.trade_cooldowns.get(symbol)
                    active_trade = self.active_orders.get(symbol)

                    if last_trade_time and (now - last_trade_time).total_seconds() < cooldown_period:
                        continue

                    if active_trade and not active_trade.get("sold", False):
                        continue

                    limit_price = round(current["close"] + 0.02, 2)
                    equity_per_trade = min(self.equity_per_trade.get(), self.available_funds)
                    quantity = int(equity_per_trade // current["close"])

                    self.last_price_change = price_change
                    self.last_rsi_change = rsi_change
                    self.last_bollinger_change = bollinger_change

                    if quantity > 0:
                        self.place_limit_order(symbol, limit_price, quantity=quantity)

                self.tree.insert("", "end", values=(
                    symbol, data["exchange"], f"${current['close']:.2f}",
                    f"{price_change:.3f}%", f"{rsi_change:.3f}%", f"{bollinger_change:.3f}%",
                    f"{current['%K']:.2f}", f"{current['%D']:.2f}", status
                ))

            except Exception as e:
                print(f"❌ Error updating {symbol}: {e}")

        # ✅ Show currently held positions (external or internal)
        try:
            positions = self.ib.positions()

            for pos in positions:
                symbol = pos.contract.symbol
                quantity = pos.position
                avg_price = pos.avgCost

                # Ensure market data is present
                if symbol not in self.market_data:
                    contract = Stock(symbol, self.exchange_var.get(), "USD")
                    try:
                        ticker = self.ib.reqMktData(contract, "", snapshot=True)
                        time.sleep(1)
                        last_price = ticker.last
                        self.market_data[symbol] = {"last": last_price}
                    except Exception as e:
                        print(f"⚠ Snapshot fetch failed for {symbol}: {e}")
                        last_price = 0.0
                else:
                    last_price = self.market_data[symbol].get("last", 0.0)

                if last_price is None or pd.isna(last_price):
                    last_price = 0.0

                unrealized_pnl = (last_price - avg_price) * quantity

                self.trades_tree.insert("", "end", values=(
                    symbol,
                    quantity,
                    f"${avg_price:.2f}",
                    f"${last_price:.2f}",
                    f"${unrealized_pnl:.2f}"
                ))

        except Exception as e:
            print(f"❌ Error fetching positions: {e}")



    def show_backtest_results_popup(self, trade_log, total_trades, total_profit, total_equity):
        print("\n📊 Optimization Backtest Results")
        print(f"🧾 Total Trades      : {total_trades}")
        print(f"💰 Total Profit      : ${total_profit:.2f}")
        print(f"📦 Total Equity Used : ${total_equity:.2f}")

        print("\n📈 Top 5 Trades:")
        for trade in trade_log[:5]:
            print(f" - {trade['Symbol']} | Buy: ${trade['Buy Price']:.2f} | Sell: ${trade['Sell Price']:.2f} | "
                f"PnL: ${trade['Profit ($)']:.2f} | Reason: {trade['Exit Reason']}")

    def start_stream(self):
        """ Start real-time market data streaming """
        if self.is_streaming:
            return

        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self.stream_quotes, daemon=True)
        self.stream_thread.start()
        print("✅ Started market data stream")

    def stop_stream(self):
        """ Stop data streaming """
        self.is_streaming = False
        print("⏹ Stopped market data stream")

    def stream_quotes(self):
        """ Fetch market quotes asynchronously using IBKR API """
        self.ib.reqMarketDataType(4)  # ✅ Use delayed market data

        symbols = self.tickers.get().split(",")
        selected_exchange = self.exchange_var.get()
        analysis_interval = self.analysis_interval.get()

        contracts = [Stock(symbol.strip(), selected_exchange, "USD") for symbol in symbols]

        async def fetch_data():
            tickers = []
            for contract in contracts:
                try:
                    ticker = self.ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
                    tickers.append(ticker)
                except Exception as e:
                    print(f"❌ Error fetching data for {contract.symbol}: {e}")
                    continue

            await asyncio.sleep(2)  # Allow time for data to populate

            while self.is_streaming:
                await asyncio.sleep(1)
                for ticker, symbol in zip(tickers, symbols):
                    if ticker.last is not None:
                        self.market_data[symbol] = {
                            "exchange": selected_exchange,
                            "analysis_interval": analysis_interval,
                            "last": ticker.last,
                            "bid": ticker.bid if ticker.bid is not None else 0,
                            "ask": ticker.ask if ticker.ask is not None else 0,
                            "volume": ticker.volume if ticker.volume is not None else 0
                        }
                    else:
                        print(f"⚠ No market data available for {symbol}")

                self.root.after(0, self.update_ui)

        def run_async_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(fetch_data())

        threading.Thread(target=run_async_task, daemon=True).start()

    def sell_position(self, symbol, price, reason="Exit"):
        """Handle selling a position"""
        contract = Stock(symbol, self.exchange_var.get(), 'USD')
        quantity = self.active_orders[symbol]['quantity']
        sell_order = Order(
            action="SELL",
            orderType="MKT",
            totalQuantity=quantity
        )
        self.ib.placeOrder(contract, sell_order)

        self.active_orders[symbol]["sold"] = True
        self.available_funds += price * quantity
        print(f"✅ SOLD {symbol} at ${price:.2f} due to {reason}. Funds recovered: ${price * quantity:.2f}")

    def toggle_force_trade(self):
        """ Toggle the force trade flag """
        current_state = self.force_trade.get()
        self.force_trade.set(not current_state)
        status = "ON" if not current_state else "OFF"
        print(f"🚀 Force Trade toggled {status}")

    def place_limit_order(self, symbol, limit_price, quantity=10, action="BUY"):
        """Place a BUY or tiered SELL limit orders for a symbol"""
        if limit_price is None or pd.isna(limit_price):
            print(f"❌ Invalid limit price for {symbol}. Order not placed.")
            return

        if action == "BUY" and quantity <= 0:
            print(f"🚫 Not enough funds to buy any shares of {symbol}. Skipping trade.")
            return

        if action == "SELL" and symbol not in self.active_orders:
            print(f"🚫 Attempted to SELL {symbol} without prior BUY. Preventing short.")
            return

        contract = Stock(symbol, self.exchange_var.get(), 'USD')

        from pytz import timezone
        eastern = timezone('US/Eastern')
        trade_time_et = datetime.now().astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S')

        self.trade_intent_log.append({
            "Time (ET)": trade_time_et,
            "Symbol": symbol,
            "Action": action,
            "Quantity": quantity,
            "Limit Price": limit_price,
            "Price Change %/min": self.last_price_change,
            "RSI Change %/min": self.last_rsi_change,
            "Bollinger % Change/min": self.last_bollinger_change
        })

        if action == "BUY":
            buy_order = Order(
                action="BUY",
                orderType="LMT",
                totalQuantity=quantity,
                lmtPrice=limit_price
            )
            self.ib.placeOrder(contract, buy_order)

            self.available_funds -= limit_price * quantity
            print(f"✅ BUY limit order placed for {symbol} at ${limit_price}, {quantity} shares")

            self.active_orders[symbol] = {
                'buy_price': limit_price,
                'quantity': quantity,
                'sold': False
            }

            for pct in range(2, 6):
                sell_price = round(limit_price * (1 + pct / 100), 2)
                sell_order = Order(
                    action="SELL",
                    orderType="LMT",
                    totalQuantity=quantity,
                    lmtPrice=sell_price
                )
                self.ib.placeOrder(contract, sell_order)
                print(f"📈 SELL limit order at +{pct}% (${sell_price}) placed for {symbol}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()