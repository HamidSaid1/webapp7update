
#!/usr/bin/env python3
"""
Simple runner script for the standalone trading application
"""

if __name__ == "__main__":
    try:
        from standalone_trading_app import TradingGUI
        print("Starting AutoTrade Plus - Standalone Trading Application...")
        app = TradingGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        input("Press Enter to exit...")
