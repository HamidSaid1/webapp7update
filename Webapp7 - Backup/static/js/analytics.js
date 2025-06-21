// Analytics JavaScript for AutoTrade Plus
// Performance tracking and system monitoring

let analyticsInterval;
let historyInterval;

// Initialize analytics page
document.addEventListener('DOMContentLoaded', function() {
    console.log('Analytics page initialized');
    
    // Start real-time updates
    updateAnalytics();
    updateTradeHistory();
    updateSystemStatus();
    
    // Set intervals for updates
    analyticsInterval = setInterval(updateAnalytics, 5000);
    historyInterval = setInterval(updateTradeHistory, 10000);
    
    // Setup event handlers
    setupEventHandlers();
    
    // Initialize Feather icons
    feather.replace();
});

// Setup event handlers
function setupEventHandlers() {
    const refreshHistoryBtn = document.getElementById('refresh-history-btn');
    
    if (refreshHistoryBtn) {
        refreshHistoryBtn.addEventListener('click', function() {
            this.innerHTML = '<i data-feather="loader" class="me-1"></i>Refreshing...';
            this.disabled = true;
            
            updateTradeHistory().finally(() => {
                this.innerHTML = '<i data-feather="refresh-cw" class="me-1"></i>Refresh';
                this.disabled = false;
                feather.replace();
            });
        });
    }
}

// Update analytics overview
function updateAnalytics() {
    fetch('/api/analytics')
        .then(response => response.json())
        .then(data => {
            // Update performance overview
            document.getElementById('analytics-total-trades').textContent = data.total_trades;
            
            // Format and color-code profit
            const profitElement = document.getElementById('analytics-total-profit');
            const profit = data.total_profit;
            profitElement.textContent = `$${profit.toFixed(2)}`;
            
            if (profit > 0) {
                profitElement.className = 'stat-value value-positive';
            } else if (profit < 0) {
                profitElement.className = 'stat-value value-negative';
            } else {
                profitElement.className = 'stat-value value-neutral';
            }
            
            // Update win rate
            document.getElementById('analytics-win-rate').textContent = `${data.win_rate.toFixed(1)}%`;
            
            // Update available funds
            document.getElementById('analytics-available-funds').textContent = `$${data.available_funds.toFixed(2)}`;
            
            // Update real-time metrics
            document.getElementById('current-equity').textContent = data.equity_per_trade.toFixed(2);
            document.getElementById('current-rsi-period').textContent = data.rsi_period;
            document.getElementById('current-bb-period').textContent = data.bb_period;
            document.getElementById('current-stoch-k-period').textContent = data.stoch_k_period;
            document.getElementById('current-stoch-d-period').textContent = data.stoch_d_period;
            document.getElementById('current-stoch-k-min').textContent = data.stoch_k_min;
            document.getElementById('current-stoch-d-min').textContent = data.stoch_d_min;
            
            // Update last rate changes
            document.getElementById('analytics-last-price-change').textContent = 
                typeof data.last_price_change === 'number' ? 
                data.last_price_change.toFixed(6) : data.last_price_change;
            
            document.getElementById('analytics-last-rsi-change').textContent = 
                typeof data.last_rsi_change === 'number' ? 
                data.last_rsi_change.toFixed(6) : data.last_rsi_change;
            
            document.getElementById('analytics-last-bollinger-change').textContent = 
                typeof data.last_bollinger_change === 'number' ? 
                data.last_bollinger_change.toFixed(6) : data.last_bollinger_change;
        })
        .catch(error => {
            console.error('Error fetching analytics:', error);
        });
}

// Update trade history table
function updateTradeHistory() {
    return fetch('/api/trade_history')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('history-body');
            
            if (!data || data.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="8" class="text-center text-muted py-4">
                            <i data-feather="inbox" class="me-2"></i>
                            No trade history available
                        </td>
                    </tr>
                `;
            } else {
                tbody.innerHTML = '';
                
                // Sort by timestamp (newest first)
                data.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                
                data.forEach(trade => {
                    const row = document.createElement('tr');
                    
                    // Format timestamps
                    const buyTime = new Date(trade.timestamp).toLocaleString();
                    const sellTime = trade.sell_timestamp ? 
                        new Date(trade.sell_timestamp).toLocaleString() : 'N/A';
                    
                    // Calculate profit/loss
                    const profit = trade.profit || 0;
                    const profitClass = profit > 0 ? 'value-positive' : 
                                       profit < 0 ? 'value-negative' : 'value-neutral';
                    
                    // Determine status
                    const status = trade.sold ? 'Closed' : 'Open';
                    const statusClass = trade.sold ? 'text-secondary' : 'text-primary';
                    
                    row.innerHTML = `
                        <td class="fw-bold">${trade.symbol}</td>
                        <td>${trade.quantity}</td>
                        <td>$${trade.buy_price.toFixed(2)}</td>
                        <td>${trade.sell_price ? '$' + trade.sell_price.toFixed(2) : 'N/A'}</td>
                        <td class="${profitClass}">$${profit.toFixed(2)}</td>
                        <td>${buyTime}</td>
                        <td>${sellTime}</td>
                        <td class="${statusClass}">${status}</td>
                    `;
                    
                    tbody.appendChild(row);
                });
            }
            
            // Re-initialize Feather icons
            feather.replace();
        })
        .catch(error => {
            console.error('Error fetching trade history:', error);
            const tbody = document.getElementById('history-body');
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center text-danger py-4">
                        <i data-feather="alert-circle" class="me-2"></i>
                        Error loading trade history
                    </td>
                </tr>
            `;
            feather.replace();
        });
}

// Update system status
function updateSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            // Update system status panel
            document.getElementById('system-connection').textContent = 
                data.connected ? 'Connected' : 'Disconnected';
            document.getElementById('system-connection').className = 
                `status-value ${data.connected ? 'text-success' : 'text-danger'}`;
            
            document.getElementById('system-streaming').textContent = 
                data.streaming ? 'Active' : 'Stopped';
            document.getElementById('system-streaming').className = 
                `status-value ${data.streaming ? 'text-info' : 'text-warning'}`;
            
            document.getElementById('system-force-trade').textContent = 
                data.force_trade_mode ? 'Enabled' : 'Disabled';
            document.getElementById('system-force-trade').className = 
                `status-value ${data.force_trade_mode ? 'text-warning' : 'text-secondary'}`;
        })
        .catch(error => {
            console.error('Error fetching system status:', error);
        });
    
    // Update active positions count
    fetch('/api/analytics')
        .then(response => response.json())
        .then(data => {
            document.getElementById('system-active-positions').textContent = data.active_positions;
            document.getElementById('system-active-positions').className = 
                `status-value ${data.active_positions > 0 ? 'text-info' : 'text-secondary'}`;
        })
        .catch(error => {
            console.error('Error fetching active positions:', error);
        });
}

// Update trading criteria status
function updateCriteriaStatus() {
    fetch('/api/market_data')
        .then(response => response.json())
        .then(data => {
            const criteriaDiv = document.getElementById('criteria-status');
            
            if (Object.keys(data).length === 0) {
                criteriaDiv.innerHTML = `
                    <div class="text-center text-muted">
                        <i data-feather="wifi-off" class="me-2"></i>
                        No market data available
                    </div>
                `;
            } else {
                criteriaDiv.innerHTML = '';
                
                Object.entries(data).forEach(([symbol, marketData]) => {
                    const criteriaItem = document.createElement('div');
                    criteriaItem.className = 'criteria-item';
                    
                    const meetsClass = marketData.meets_criteria ? 'criteria-met' : 'criteria-not-met';
                    const meetsText = marketData.meets_criteria ? 'MEETS ALL' : 'INCOMPLETE';
                    
                    criteriaItem.innerHTML = `
                        <span class="criteria-label">${symbol} Criteria:</span>
                        <span class="criteria-value ${meetsClass}">${meetsText}</span>
                    `;
                    
                    criteriaDiv.appendChild(criteriaItem);
                });
            }
            
            feather.replace();
        })
        .catch(error => {
            console.error('Error fetching criteria status:', error);
        });
}

// Update criteria status every 5 seconds
setInterval(updateCriteriaStatus, 5000);

// Initial load of criteria status
updateCriteriaStatus();

// Update system status every 5 seconds
setInterval(updateSystemStatus, 5000);

// Initial load of system status
updateSystemStatus();

// Cleanup intervals when page unloads
window.addEventListener('beforeunload', function() {
    if (analyticsInterval) clearInterval(analyticsInterval);
    if (historyInterval) clearInterval(historyInterval);
});

// Export function for manual refresh
window.refreshAnalytics = function() {
    updateAnalytics();
    updateTradeHistory();
    updateSystemStatus();
    updateCriteriaStatus();
};

// Performance monitoring (optional feature)
function trackPerformance() {
    const startTime = Date.now();
    
    return {
        mark: function(label) {
            const elapsed = Date.now() - startTime;
            console.log(`Analytics Performance - ${label}: ${elapsed}ms`);
        }
    };
}

// Add performance tracking to major functions
const originalUpdateAnalytics = updateAnalytics;
updateAnalytics = function() {
    const perf = trackPerformance();
    return originalUpdateAnalytics().then(() => {
        perf.mark('Analytics Update Complete');
    });
};
