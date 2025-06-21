// Settings JavaScript for AutoTrade Plus
// Parameter configuration with exact reserve_file.py values

// Default parameters from reserve_file.py
const defaultParameters = {
    tickers: 'ABTS',
    data_points: 15,
    chart_interval: '5 mins',
    analysis_interval: 1,
    rsi_period: 2,
    bb_period: 2,
    stoch_k_period: 2,
    stoch_d_period: 2,
    stoch_k_min: 50.0,
    stoch_d_min: 40.0,
    duration_str: '1 D',
    equity_per_trade: 100.0,
    exchange_var: 'SMART',
    hard_stop_loss: 0.03
};

// Initialize settings page
document.addEventListener('DOMContentLoaded', function() {
    console.log('Settings page initialized');

    // Load current parameters
    loadCurrentParameters();

    // Setup form handlers
    setupFormHandlers();

    // Initialize Feather icons
    feather.replace();
});

// Load current parameters from API
function loadCurrentParameters() {
    fetch('/api/parameters')
        .then(response => response.json())
        .then(data => {
            // Populate form fields
            Object.entries(data).forEach(([key, value]) => {
                const element = document.getElementById(key);
                if (element) {
                    element.value = value;
                }
            });

            // Update settings summary
            updateSettingsSummary(data);
        })
        .catch(error => {
            console.error('Error loading parameters:', error);
            showError('Failed to load current parameters');
        });
}

// Update settings summary panel
function updateSettingsSummary(params) {
    const summaryDiv = document.getElementById('settings-summary');

    summaryDiv.innerHTML = `
        <div class="setting-item">
            <span class="setting-label">Tickers:</span>
            <span class="setting-value">${params.tickers}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Exchange:</span>
            <span class="setting-value">${params.exchange_var}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Chart Interval:</span>
            <span class="setting-value">${params.chart_interval}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Analysis Interval:</span>
            <span class="setting-value">${params.analysis_interval}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">RSI Period:</span>
            <span class="setting-value">${params.rsi_period}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Bollinger Period:</span>
            <span class="setting-value">${params.bb_period}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Stoch %K Period:</span>
            <span class="setting-value">${params.stoch_k_period}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Stoch %D Period:</span>
            <span class="setting-value">${params.stoch_d_period}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Stoch %K Min:</span>
            <span class="setting-value">${params.stoch_k_min}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Stoch %D Min:</span>
            <span class="setting-value">${params.stoch_d_min}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Equity Per Trade:</span>
            <span class="setting-value">$${params.equity_per_trade}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Hard Stop Loss:</span>
            <span class="setting-value">${(params.hard_stop_loss * 100).toFixed(1)}%</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Duration:</span>
            <span class="setting-value">${params.duration_str}</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">Data Points:</span>
            <span class="setting-value">${params.data_points}</span>
        </div>
    `;
}

// Setup form event handlers
function setupFormHandlers() {
    const form = document.getElementById('settings-form');
    const resetBtn = document.getElementById('reset-btn');

    // Form submission handler
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        saveParameters();
    });

    // Reset button handler
    resetBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to reset all parameters to default values from reserve_file.py?')) {
            resetToDefaults();
        }
    });

    // Real-time validation
    setupValidation();
}

// Setup form validation
function setupValidation() {
    const form = document.getElementById('settings-form');
    const numberInputs = form.querySelectorAll('input[type="number"]');
    const textInputs = form.querySelectorAll('input[type="text"]');
    const selectInputs = form.querySelectorAll('select');

    // Validate number inputs
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            validateNumberInput(this);
        });

        input.addEventListener('blur', function() {
            validateNumberInput(this);
        });

        // Initial validation
        validateNumberInput(input);
    });

    // Validate text inputs
    textInputs.forEach(input => {
        input.addEventListener('input', function() {
            validateTextInput(this);
        });

        input.addEventListener('blur', function() {
            validateTextInput(this);
        });

        // Initial validation
        validateTextInput(input);
    });

    // Validate select inputs
    selectInputs.forEach(select => {
        select.addEventListener('change', function() {
            validateSelectInput(this);
        });

        // Initial validation
        validateSelectInput(select);
    });
}

// Validate number input with field-specific rules
function validateNumberInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min) || 0;
    const max = parseFloat(input.max) || Infinity;

    // Remove previous validation classes
    input.classList.remove('is-valid', 'is-invalid');

    // Field-specific validation rules
    const fieldRules = {
        'analysis_interval': { min: 1, max: 60 },
        'rsi_period': { min: 1, max: 50 },
        'bb_period': { min: 1, max: 50 },
        'stoch_k_period': { min: 1, max: 50 },
        'stoch_d_period': { min: 1, max: 50 },
        'stoch_k_min': { min: 0, max: 100 },
        'stoch_d_min': { min: 0, max: 100 },
        'equity_per_trade': { min: 1, max: 100000 },
        'hard_stop_loss': { min: 0.001, max: 1 },
        'data_points': { min: 5, max: 1000 }
    };

    // Apply field-specific rules
    const rules = fieldRules[input.id];
    if (rules) {
        const fieldMin = rules.min;
        const fieldMax = rules.max;

        if (isNaN(value) || value < fieldMin || value > fieldMax) {
            input.classList.add('is-invalid');
            input.title = `Value must be between ${fieldMin} and ${fieldMax}`;
            return false;
        }
    } else {
        // Default validation
        if (isNaN(value) || value < min || value > max) {
            input.classList.add('is-invalid');
            input.title = `Value must be between ${min} and ${max}`;
            return false;
        }
    }

    input.classList.add('is-valid');
    input.title = '';
    return true;
}

// Validate text input with enhanced patterns
function validateTextInput(input) {
    const value = input.value.trim();

    // Remove previous validation classes
    input.classList.remove('is-valid', 'is-invalid');

    // Field-specific text validation
    if (input.id === 'tickers') {
        // Ticker symbols: letters, numbers, commas, spaces
        const tickerPattern = /^[A-Za-z0-9,\s]+$/;
        if (value.length > 0 && tickerPattern.test(value)) {
            // Additional check: ensure no empty symbols
            const symbols = value.split(',').map(s => s.trim()).filter(s => s.length > 0);
            if (symbols.length > 0 && symbols.every(s => /^[A-Za-z0-9]+$/.test(s))) {
                input.classList.add('is-valid');
                input.title = `Valid symbols: ${symbols.join(', ')}`;
                return true;
            }
        }
        input.classList.add('is-invalid');
        input.title = 'Enter valid ticker symbols (e.g., AAPL,MSFT,GOOGL)';
        return false;
    }

    if (input.id === 'duration_str') {
        // Duration format: number + space + unit (e.g., "1 D", "5 mins")
        const durationPattern = /^\d+\s+(D|H|mins?|secs?|W|M|Y)$/i;
        if (value.length > 0 && durationPattern.test(value)) {
            input.classList.add('is-valid');
            input.title = 'Valid duration format';
            return true;
        }
        input.classList.add('is-invalid');
        input.title = 'Format: number + space + unit (e.g., "1 D", "30 mins")';
        return false;
    }

    // Default text validation - require non-empty
    if (value.length > 0) {
        input.classList.add('is-valid');
        input.title = '';
        return true;
    } else {
        input.classList.add('is-invalid');
        input.title = 'This field is required';
        return false;
    }
}

// Validate select input
function validateSelectInput(select) {
    const value = select.value;

    // Remove previous validation classes
    select.classList.remove('is-valid', 'is-invalid');

    // Check if value is selected
    if (value && value !== '') {
        select.classList.add('is-valid');
        return true;
    } else {
        select.classList.add('is-invalid');
        return false;
    }
}

// Save parameters to server
function saveParameters() {
    const form = document.getElementById('settings-form');
    const formData = new FormData(form);
    const parameters = {};

    // Convert form data to object
    for (let [key, value] of formData.entries()) {
        // Convert numeric values
        const input = document.getElementById(key);
        if (input && input.type === 'number') {
            parameters[key] = parseFloat(value);
        } else {
            parameters[key] = value;
        }
    }

    // Validate all parameters
    if (!validateAllParameters(parameters)) {
        showError('Please fix validation errors before saving');
        return;
    }

    // Show loading state
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i data-feather="loader" class="me-1"></i>Saving...';
    submitBtn.disabled = true;

    // Send to server
    fetch('/api/parameters', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(parameters)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showSuccess('Parameters saved successfully');
            updateSettingsSummary(parameters);
        } else {
            showError(`Error saving parameters: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error saving parameters:', error);
        showError('Network error saving parameters');
    })
    .finally(() => {
        // Restore button state
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        feather.replace();
    });
}

// Validate all parameters
function validateAllParameters(params) {
    let isValid = true;

    // Validate required fields
    const requiredFields = [
        'tickers', 'equity_per_trade', 'rsi_period', 'bb_period',
        'stoch_k_period', 'stoch_d_period', 'stoch_k_min', 'stoch_d_min'
    ];

    requiredFields.forEach(field => {
        if (!params[field] || (typeof params[field] === 'number' && isNaN(params[field]))) {
            isValid = false;
            const input = document.getElementById(field);
            if (input) input.classList.add('is-invalid');
        }
    });

    // Validate ranges
    const rangeValidations = [
        { field: 'equity_per_trade', min: 1 },
        { field: 'rsi_period', min: 1, max: 50 },
        { field: 'bb_period', min: 1, max: 50 },
        { field: 'stoch_k_period', min: 1, max: 50 },
        { field: 'stoch_d_period', min: 1, max: 50 },
        { field: 'stoch_k_min', min: 0, max: 100 },
        { field: 'stoch_d_min', min: 0, max: 100 },
        { field: 'hard_stop_loss', min: 0.001, max: 0.5 }
    ];

    rangeValidations.forEach(validation => {
        const value = params[validation.field];
        if (value < validation.min || (validation.max && value > validation.max)) {
            isValid = false;
            const input = document.getElementById(validation.field);
            if (input) input.classList.add('is-invalid');
        }
    });

    return isValid;
}

// Reset to default parameters
function resetToDefaults() {
    // Populate form with defaults
    Object.entries(defaultParameters).forEach(([key, value]) => {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
            element.classList.remove('is-valid', 'is-invalid');
        }
    });

    // Update summary
    updateSettingsSummary(defaultParameters);

    showSuccess('Parameters reset to reserve_file.py defaults');
}

// Show success message
function showSuccess(message) {
    // Remove existing alerts
    removeAlerts();

    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show';
    alert.innerHTML = `
        <i data-feather="check-circle" class="me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert at top of card body
    const cardBody = document.querySelector('.card-body');
    cardBody.insertBefore(alert, cardBody.firstChild);

    feather.replace();

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

// Show error message
function showError(message) {
    // Remove existing alerts
    removeAlerts();

    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = `
        <i data-feather="alert-circle" class="me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert at top of card body
    const cardBody = document.querySelector('.card-body');
    cardBody.insertBefore(alert, cardBody.firstChild);

    feather.replace();

    // Auto-remove after 8 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 8000);
}

// Remove existing alerts
function removeAlerts() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => alert.remove());
}

// Start streaming
function startStreaming() {
    const btn = document.getElementById('start-streaming-btn');
    const originalText = btn.innerHTML;

    btn.innerHTML = '<i data-feather="loader" class="me-1"></i>Starting...';
    btn.disabled = true;

    fetch('/api/start_streaming', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showSuccess(data.message + ' - Check the Dashboard for live data');
                updateSystemStatus();
            } else {
                showError(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Start streaming error:', error);
            showError('Failed to start streaming');
        })
        .finally(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
            feather.replace();
        });
}

// Stop streaming
function stopStreaming() {
    const btn = document.getElementById('stop-streaming-btn');
    const originalText = btn.innerHTML;

    btn.innerHTML = '<i data-feather="loader" class="me-1"></i>Stopping...';
    btn.disabled = true;

    fetch('/api/stop_streaming', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showSuccess(data.message);
                updateSystemStatus();
            } else {
                showError(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Stop streaming error:', error);
            showError('Failed to stop streaming');
        })
        .finally(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
            feather.replace();
        });
}

// Real-time parameter updates (optional feature)
function enableRealTimeUpdates() {
    const form = document.getElementById('settings-form');
    const inputs = form.querySelectorAll('input, select');

    inputs.forEach(input => {
        input.addEventListener('change', function() {
            // Debounce updates
            clearTimeout(this.updateTimeout);
            this.updateTimeout = setTimeout(() => {
                let isValid = false;
                if (this.type === 'number') {
                    isValid = validateNumberInput(this);
                } else if (this.type === 'text') {
                    isValid = validateTextInput(this);
                } else if (this.tagName === 'SELECT') {
                    isValid = validateSelectInput(this);
                }

                if (isValid) {
                    // Update summary in real-time
                    const currentParams = getCurrentFormParameters();
                    updateSettingsSummary(currentParams);
                }
            }, 500);
        });
    });
}

// Get current form parameters
function getCurrentFormParameters() {
    const form = document.getElementById('settings-form');
    const formData = new FormData(form);
    const parameters = {};

    for (let [key, value] of formData.entries()) {
        const input = document.getElementById(key);
        if (input && input.type === 'number') {
            parameters[key] = parseFloat(value);
        } else {
            parameters[key] = value;
        }
    }

    return parameters;
}

// Get current form parameters
function getCurrentFormParameters() {
    const form = document.getElementById('settings-form');
    const formData = new FormData(form);
    const parameters = {};

    for (let [key, value] of formData.entries()) {
        const input = document.getElementById(key);
        if (input && input.type === 'number') {
            parameters[key] = parseFloat(value) || 0;
        } else {
            parameters[key] = value;
        }
    }

    return parameters;
}

// Enable real-time updates on page load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(enableRealTimeUpdates, 1000);
});

function validateForm() {
    const form = document.getElementById('settingsForm');
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;

    inputs.forEach(input => {
        const value = input.value.trim();

        if (!value) {
            input.classList.add('is-invalid');
            input.classList.remove('is-valid');
            isValid = false;
        } else {
            // Additional validation for specific fields
            if (input.type === 'number') {
                const num = parseFloat(value);
                if (isNaN(num) || num <= 0) {
                    input.classList.add('is-invalid');
                    input.classList.remove('is-valid');
                    isValid = false;
                } else {
                    input.classList.remove('is-invalid');
                    input.classList.add('is-valid');
                }
            } else {
                input.classList.remove('is-invalid');
                input.classList.add('is-valid');
            }
        }
    });

    // Remove validation classes after a short delay to prevent flickering
    setTimeout(() => {
        inputs.forEach(input => {
            if (input.classList.contains('is-valid')) {
                input.classList.remove('is-valid');
            }
        });
    }, 2000);

    return isValid;
}

// Clear validation states on input focus
document.addEventListener('DOMContentLoaded', function() {
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.classList.remove('is-invalid', 'is-valid');
        });

        input.addEventListener('input', function() {
            if (this.classList.contains('is-invalid')) {
                this.classList.remove('is-invalid');
            }
        });
    });
});