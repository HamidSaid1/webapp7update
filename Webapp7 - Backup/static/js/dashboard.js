// Dashboard JavaScript for AutoTrade Plus
// Real-time market data and trading interface

let marketDataInterval;
let analyticsInterval;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');

    // Start real-time updates
    updateMarketData();
    updateAnalytics();

    // Set intervals for updates
    marketDataInterval = setInterval(updateMarketData, 5000);
    analyticsInterval = setInterval(updateAnalytics, 5000);

    // Initialize Feather icons
    feather.replace();
});

// Update market data table
function updateMarketData() {
    fetch('/api/market_data')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('market-data-body');

            if (Object.keys(data).length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="11" class="text-center text-muted py-4">
                            <i data-feather="wifi-off" class="me-2"></i>
                            No market data available. Please connect and start streaming.
                        </td>
                    </tr>
                `;
            } else {
                tbody.innerHTML = '';

                Object.entries(data).forEach(([symbol, marketData]) => {
                    const row = document.createElement('tr');

                    // Determine status class
                    const statusClass = marketData.meets_criteria ? 'status-meets-criteria' : 'status-no-criteria';
                    const statusText = marketData.meets_criteria ? 'MEETS CRITERIA' : 'No criteria';

                    // Format numbers with proper precision
                    const formatNumber = (value, decimals = 2) => {
                        if (typeof value === 'number') {
                            return value.toFixed(decimals);
                        }
                        return value || 'N/A';
                    };

                    row.innerHTML = `
                        <td class="fw-bold">${symbol}</td>
                        <td>${marketData.exchange}</td>
                        <td class="fw-bold">$${formatNumber(marketData.last, 2)}</td>
                        <td>${formatNumber(marketData.rsi, 2)}</td>
                        <td>${formatNumber(marketData.bollinger_pct, 2)}%</td>
                        <td>${formatNumber(marketData.stoch_k, 2)}</td>
                        <td>${formatNumber(marketData.stoch_d, 2)}</td>
                        <td>${formatNumber(marketData.price_rate, 6)}</td>
                        <td>${formatNumber(marketData.rsi_rate, 6)}</td>
                        <td>${formatNumber(marketData.bollinger_rate, 6)}</td>
                        <td class="${statusClass}">${statusText}</td>
                    `;

                    tbody.appendChild(row);
                });
            }

            // Update last update time
            document.getElementById('last-update').textContent = 
                `Last updated: ${new Date().toLocaleTimeString()}`;

            // Re-initialize Feather icons
            feather.replace();
        })
        .catch(error => {
            console.error('Error fetching market data:', error);
        });
}

// Update analytics and stats
function updateAnalytics() {
    fetch('/api/analytics')
        .then(response => response.json())
        .then(data => {
            // Update quick stats
            document.getElementById('available-funds').textContent = `$${data.available_funds.toFixed(2)}`;
            document.getElementById('active-positions').textContent = data.active_positions;
            document.getElementById('total-trades').textContent = data.total_trades;

            // Format profit with color coding
            const profitElement = document.getElementById('total-profit');
            const profit = data.total_profit;
            profitElement.textContent = `$${profit.toFixed(2)}`;

            if (profit > 0) {
                profitElement.className = 'stat-value value-positive';
            } else if (profit < 0) {
                profitElement.className = 'stat-value value-negative';
            } else {
                profitElement.className = 'stat-value value-neutral';
            }

            // Update last changes
            document.getElementById('last-price-change').textContent = 
                typeof data.last_price_change === 'number' ? 
                data.last_price_change.toFixed(6) : data.last_price_change;

            document.getElementById('last-rsi-change').textContent = 
                typeof data.last_rsi_change === 'number' ? 
                data.last_rsi_change.toFixed(6) : data.last_rsi_change;

            document.getElementById('last-bollinger-change').textContent = 
                typeof data.last_bollinger_change === 'number' ? 
                data.last_bollinger_change.toFixed(6) : data.last_bollinger_change;
        })
        .catch(error => {
            console.error('Error fetching analytics:', error);
        });
}

// Update open trades table
function updateOpenTrades() {
    fetch('/api/active_orders')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('trades-body');

            if (Object.keys(data).length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="7" class="text-center text-muted py-4">
                            <i data-feather="inbox" class="me-2"></i>
                            No open trades
                        </td>
                    </tr>
                `;
            } else {
                tbody.innerHTML = '';

                // Also fetch current market data for P&L calculation
                fetch('/api/market_data')
                    .then(response => response.json())
                    .then(marketData => {
                        Object.entries(data).forEach(([symbol, trade]) => {
                            if (!trade.sold) {
                                const row = document.createElement('tr');

                                const currentPrice = marketData[symbol]?.last || trade.buy_price;
                                const unrealizedPnL = (currentPrice - trade.buy_price) * trade.quantity;

                                // Format timestamp
                                const timestamp = new Date(trade.timestamp).toLocaleString();

                                row.innerHTML = `
                                    <td class="fw-bold">${symbol}</td>
                                    <td>${trade.quantity}</td>
                                    <td>$${trade.buy_price.toFixed(2)}</td>
                                    <td>$${currentPrice.toFixed(2)}</td>
                                    <td class="${unrealizedPnL >= 0 ? 'value-positive' : 'value-negative'}">
                                        $${unrealizedPnL.toFixed(2)}
                                    </td>
                                    <td>${timestamp}</td>
                                    <td>
                                        <button class="btn btn-outline-danger btn-sm" onclick="sellPosition('${symbol}')">
                                            <i data-feather="x" class="me-1"></i>Sell
                                        </button>
                                    </td>
                                `;

                                tbody.appendChild(row);
                            }
                        });

                        // Re-initialize Feather icons
                        feather.replace();
                    });
            }
        })
        .catch(error => {
            console.error('Error fetching active orders:', error);
        });
}

// Sell position function (placeholder for now)
function sellPosition(symbol) {
    if (confirm(`Are you sure you want to sell all ${symbol} positions?`)) {
        // TODO: Implement sell functionality
        alert(`Sell functionality for ${symbol} will be implemented in the next version.`);
    }
}

// Update open trades every 5 seconds
setInterval(updateOpenTrades, 5000);

// Initial load of open trades
updateOpenTrades();

// Cleanup intervals when page unloads
window.addEventListener('beforeunload', function() {
    if (marketDataInterval) clearInterval(marketDataInterval);
    if (analyticsInterval) clearInterval(analyticsInterval);
});

// Force trade toggle
    $('#forceTradeBtn').click(function() {
        $.post('/api/toggle_force_trade')
            .done(function(data) {
                if (data.success) {
                    showAlert(data.message, 'info');
                    updateForceTradeStatus(data.force_trade_enabled);
                    if (data.force_trade_enabled) {
                        showAlert('⚠️ Force Trade Mode Enabled!\nThis mode bypasses safety checks and may increase risk.\nUse with caution!', 'warning');
                    }
                } else {
                    showAlert('Error: ' + data.error, 'danger');
                }
            })
            .fail(function() {
                showAlert('Failed to toggle force trade mode', 'danger');
            });
    });

    // Force buy button
    $('#forceBuyBtn').click(function() {
        executeForceTrace('BUY');
    });

    // Force sell button
    $('#forceSellBtn').click(function() {
        executeForceTrace('SELL');
    });

    function executeForceTrace(action) {
        const symbol = $('#forceTradeSymbol').val();
        if (!symbol) {
            showAlert('Please select a stock symbol', 'warning');
            return;
        }

        $.post('/api/force_trade', {
            symbol: symbol,
            action: action
        })
        .done(function(data) {
            if (data.success) {
                showAlert(data.message, 'success');
                updateTables(); // Refresh data
            } else {
                showAlert('Force ' + action + ' failed: ' + data.error, 'danger');
            }
        })
        .fail(function() {
            showAlert('Failed to execute force ' + action, 'danger');
        });
    }

    function updateForceTradeStatus(enabled) {
        const statusText = 'Force trade mode: ' + (enabled ? 'Enabled' : 'Disabled');
        $('#forceTradeStatus').text(statusText);
        $('#forceBuyBtn, #forceSellBtn').prop('disabled', !enabled);
    }

marketDataTableBody.empty();

            if (data && Object.keys(data).length > 0) {
                // Update force trade symbol dropdown
                const forceTradeSymbol = $('#forceTradeSymbol');
                const currentValue = forceTradeSymbol.val();
                forceTradeSymbol.empty().append('<option value="">Select Symbol</option>');

                $.each(data, function(symbol, info) {
                    const row = `
                        <tr>
                            <td>${symbol}</td>
                            <td>$${parseFloat(info.last || 0).toFixed(2)}</td>
                            <td>${parseFloat(info.rsi || 0).toFixed(1)}</td>
                            <td>${parseFloat(info.bollinger_pct || 0).toFixed(1)}%</td>
                            <td>${parseFloat(info.stoch_k || 0).toFixed(1)}</td>
                            <td>${parseFloat(info.stoch_d || 0).toFixed(1)}</td>
                            <td>${parseInt(info.volume || 0).toLocaleString()}</td>
                            <td>${info.status || 'N/A'}</td>
                            <td>
                                <span class="badge ${info.meets_criteria ? 'badge-success' : 'badge-secondary'}">
                                    ${info.meets_criteria ? 'Yes' : 'No'}
                                </span>
                            </td>
                        </tr>
                    `;
                    marketDataTableBody.append(row);

                    // Add to symbol dropdown
                    const selected = (currentValue === symbol || (!currentValue && symbol === Object.keys(data)[0])) ? 'selected' : '';
                    forceTradeSymbol.append(`<option value="${symbol}" ${selected}>${symbol}</option>`);
                });
            } else {
                marketDataTableBody.append('<tr><td colspan="9" class="text-center">No market data available</td></tr>');
            }