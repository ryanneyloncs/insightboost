/**
 * InsightBoost - Main Application JavaScript
 * Core functionality, utilities, and global event handlers
 */

(function() {
    'use strict';

    // ============================================
    // Configuration
    // ============================================
    const CONFIG = {
        API_BASE: '/api/v1',
        TOAST_DURATION: 5000,
        DEBOUNCE_DELAY: 300,
        MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
        SUPPORTED_FORMATS: ['.csv', '.xlsx', '.xls', '.json', '.parquet'],
    };

    // ============================================
    // Global State
    // ============================================
    window.InsightBoost = window.InsightBoost || {
        state: {
            currentDataset: null,
            datasets: [],
            insights: [],
            visualizations: [],
            sessions: [],
            isLoading: false,
        },
        config: CONFIG,
    };

    // ============================================
    // API Client
    // ============================================
    const API = {
        /**
         * Make an API request
         * @param {string} endpoint - API endpoint
         * @param {object} options - Fetch options
         * @returns {Promise<object>} Response data
         */
        async request(endpoint, options = {}) {
            const url = endpoint.startsWith('http') ? endpoint : `${CONFIG.API_BASE}${endpoint}`;
            
            const defaultOptions = {
                headers: {
                    'Content-Type': 'application/json',
                },
            };

            const mergedOptions = {
                ...defaultOptions,
                ...options,
                headers: {
                    ...defaultOptions.headers,
                    ...options.headers,
                },
            };

            // Don't set Content-Type for FormData
            if (options.body instanceof FormData) {
                delete mergedOptions.headers['Content-Type'];
            }

            try {
                const response = await fetch(url, mergedOptions);
                const data = await response.json();

                if (!response.ok) {
                    throw new APIError(data.message || 'Request failed', response.status, data);
                }

                return data;
            } catch (error) {
                if (error instanceof APIError) {
                    throw error;
                }
                throw new APIError('Network error', 0, { originalError: error.message });
            }
        },

        // Convenience methods
        get: (endpoint) => API.request(endpoint, { method: 'GET' }),
        post: (endpoint, data) => API.request(endpoint, { method: 'POST', body: JSON.stringify(data) }),
        put: (endpoint, data) => API.request(endpoint, { method: 'PUT', body: JSON.stringify(data) }),
        delete: (endpoint) => API.request(endpoint, { method: 'DELETE' }),
        upload: (endpoint, formData) => API.request(endpoint, { method: 'POST', body: formData }),
    };

    // Custom API Error
    class APIError extends Error {
        constructor(message, status, data) {
            super(message);
            this.name = 'APIError';
            this.status = status;
            this.data = data;
        }
    }

    // Expose API globally
    window.InsightBoost.API = API;

    // ============================================
    // Toast Notifications
    // ============================================
    const Toast = {
        container: null,

        init() {
            this.container = document.getElementById('toast-container');
            if (!this.container) {
                this.container = document.createElement('div');
                this.container.id = 'toast-container';
                this.container.className = 'fixed bottom-4 right-4 z-50 space-y-2';
                document.body.appendChild(this.container);
            }
        },

        /**
         * Show a toast notification
         * @param {string} message - Toast message
         * @param {string} type - Toast type (success, error, warning, info)
         * @param {number} duration - Duration in ms
         */
        show(message, type = 'info', duration = CONFIG.TOAST_DURATION) {
            if (!this.container) this.init();

            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            
            const icons = {
                success: `<svg class="h-5 w-5 text-green-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>`,
                error: `<svg class="h-5 w-5 text-red-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>`,
                warning: `<svg class="h-5 w-5 text-yellow-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>`,
                info: `<svg class="h-5 w-5 text-blue-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>`,
            };

            toast.innerHTML = `
                ${icons[type] || icons.info}
                <span class="text-sm text-gray-700 flex-1">${message}</span>
                <button onclick="this.parentElement.remove()" class="ml-3 text-gray-400 hover:text-gray-600">
                    <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            `;

            this.container.appendChild(toast);

            // Auto-remove after duration
            if (duration > 0) {
                setTimeout(() => {
                    toast.classList.add('removing');
                    setTimeout(() => toast.remove(), 300);
                }, duration);
            }

            return toast;
        },

        success: (message, duration) => Toast.show(message, 'success', duration),
        error: (message, duration) => Toast.show(message, 'error', duration),
        warning: (message, duration) => Toast.show(message, 'warning', duration),
        info: (message, duration) => Toast.show(message, 'info', duration),
    };

    // Global toast function
    window.showToast = Toast.show.bind(Toast);

    // ============================================
    // Loading Overlay
    // ============================================
    const Loading = {
        overlay: null,
        messageEl: null,

        init() {
            this.overlay = document.getElementById('loading-overlay');
            this.messageEl = document.getElementById('loading-message');
        },

        show(message = 'Processing...') {
            if (!this.overlay) this.init();
            if (this.overlay) {
                this.overlay.classList.remove('hidden');
                if (this.messageEl) {
                    this.messageEl.textContent = message;
                }
            }
            window.InsightBoost.state.isLoading = true;
        },

        hide() {
            if (!this.overlay) this.init();
            if (this.overlay) {
                this.overlay.classList.add('hidden');
            }
            window.InsightBoost.state.isLoading = false;
        },
    };

    // Global loading functions
    window.showLoading = Loading.show.bind(Loading);
    window.hideLoading = Loading.hide.bind(Loading);

    // ============================================
    // Utility Functions
    // ============================================
    const Utils = {
        /**
         * Format bytes to human readable string
         * @param {number} bytes - Number of bytes
         * @returns {string} Formatted string
         */
        formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },

        /**
         * Format number with thousand separators
         * @param {number} num - Number to format
         * @returns {string} Formatted string
         */
        formatNumber(num) {
            return num.toLocaleString();
        },

        /**
         * Format date to locale string
         * @param {string|Date} date - Date to format
         * @returns {string} Formatted date
         */
        formatDate(date) {
            return new Date(date).toLocaleDateString();
        },

        /**
         * Format time to locale string
         * @param {string|Date} date - Date to format
         * @returns {string} Formatted time
         */
        formatTime(date) {
            return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        },

        /**
         * Format relative time (e.g., "2 hours ago")
         * @param {string|Date} date - Date to format
         * @returns {string} Relative time string
         */
        formatRelativeTime(date) {
            const now = new Date();
            const then = new Date(date);
            const diff = now - then;

            const seconds = Math.floor(diff / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            const days = Math.floor(hours / 24);

            if (days > 7) return this.formatDate(date);
            if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
            if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
            if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
            return 'Just now';
        },

        /**
         * Debounce a function
         * @param {Function} func - Function to debounce
         * @param {number} wait - Wait time in ms
         * @returns {Function} Debounced function
         */
        debounce(func, wait = CONFIG.DEBOUNCE_DELAY) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        /**
         * Throttle a function
         * @param {Function} func - Function to throttle
         * @param {number} limit - Limit in ms
         * @returns {Function} Throttled function
         */
        throttle(func, limit) {
            let inThrottle;
            return function(...args) {
                if (!inThrottle) {
                    func.apply(this, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },

        /**
         * Generate a random ID
         * @param {number} length - ID length
         * @returns {string} Random ID
         */
        generateId(length = 8) {
            return Math.random().toString(36).substr(2, length);
        },

        /**
         * Deep clone an object
         * @param {object} obj - Object to clone
         * @returns {object} Cloned object
         */
        deepClone(obj) {
            return JSON.parse(JSON.stringify(obj));
        },

        /**
         * Check if file format is supported
         * @param {string} filename - File name
         * @returns {boolean} Is supported
         */
        isSupportedFormat(filename) {
            const ext = '.' + filename.split('.').pop().toLowerCase();
            return CONFIG.SUPPORTED_FORMATS.includes(ext);
        },

        /**
         * Escape HTML to prevent XSS
         * @param {string} str - String to escape
         * @returns {string} Escaped string
         */
        escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        },

        /**
         * Copy text to clipboard
         * @param {string} text - Text to copy
         * @returns {Promise<void>}
         */
        async copyToClipboard(text) {
            try {
                await navigator.clipboard.writeText(text);
                Toast.success('Copied to clipboard');
            } catch (err) {
                Toast.error('Failed to copy');
            }
        },

        /**
         * Download a blob as a file
         * @param {Blob} blob - Blob to download
         * @param {string} filename - File name
         */
        downloadBlob(blob, filename) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        },

        /**
         * Download data as JSON file
         * @param {object} data - Data to download
         * @param {string} filename - File name
         */
        downloadJSON(data, filename) {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            this.downloadBlob(blob, filename);
        },
    };

    // Expose utilities globally
    window.InsightBoost.Utils = Utils;
    window.formatBytes = Utils.formatBytes;

    // ============================================
    // Event Handlers
    // ============================================
    
    // Initialize on DOM ready
    document.addEventListener('DOMContentLoaded', function() {
        Toast.init();
        Loading.init();
        
        // Initialize any components
        initializeDropdowns();
        initializeTooltips();
        initializeKeyboardShortcuts();
    });

    // Initialize dropdowns
    function initializeDropdowns() {
        document.addEventListener('click', function(e) {
            // Close all dropdowns when clicking outside
            if (!e.target.closest('[data-dropdown]')) {
                document.querySelectorAll('.dropdown-menu.active').forEach(menu => {
                    menu.classList.remove('active');
                });
            }
        });
    }

    // Initialize tooltips
    function initializeTooltips() {
        // Tooltips are handled via CSS [data-tooltip] attribute
    }

    // Initialize keyboard shortcuts
    function initializeKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Escape to close modals
            if (e.key === 'Escape') {
                const modal = document.querySelector('.modal-overlay.active, [id$="-modal"]:not(.hidden)');
                if (modal) {
                    modal.classList.add('hidden');
                    e.preventDefault();
                }
            }

            // Ctrl/Cmd + K for quick search (if implemented)
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                const searchInput = document.getElementById('global-search');
                if (searchInput) {
                    searchInput.focus();
                    e.preventDefault();
                }
            }
        });
    }

    // ============================================
    // Global Event Bus
    // ============================================
    const EventBus = {
        events: {},

        on(event, callback) {
            if (!this.events[event]) {
                this.events[event] = [];
            }
            this.events[event].push(callback);
            return () => this.off(event, callback);
        },

        off(event, callback) {
            if (!this.events[event]) return;
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        },

        emit(event, data) {
            if (!this.events[event]) return;
            this.events[event].forEach(callback => callback(data));
        },
    };

    window.InsightBoost.EventBus = EventBus;

    // ============================================
    // Form Validation
    // ============================================
    const Validation = {
        /**
         * Validate a form field
         * @param {HTMLInputElement} field - Form field
         * @returns {boolean} Is valid
         */
        validateField(field) {
            const value = field.value.trim();
            let isValid = true;
            let message = '';

            // Required validation
            if (field.required && !value) {
                isValid = false;
                message = 'This field is required';
            }

            // Email validation
            if (isValid && field.type === 'email' && value) {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(value)) {
                    isValid = false;
                    message = 'Please enter a valid email';
                }
            }

            // Min length validation
            if (isValid && field.minLength && value.length < field.minLength) {
                isValid = false;
                message = `Minimum ${field.minLength} characters required`;
            }

            // Max length validation
            if (isValid && field.maxLength && value.length > field.maxLength) {
                isValid = false;
                message = `Maximum ${field.maxLength} characters allowed`;
            }

            // Update field UI
            this.updateFieldUI(field, isValid, message);
            return isValid;
        },

        /**
         * Update field UI based on validation
         * @param {HTMLInputElement} field - Form field
         * @param {boolean} isValid - Is valid
         * @param {string} message - Error message
         */
        updateFieldUI(field, isValid, message) {
            const errorEl = field.parentElement.querySelector('.field-error');
            
            if (isValid) {
                field.classList.remove('border-red-500');
                field.classList.add('border-gray-300');
                if (errorEl) errorEl.remove();
            } else {
                field.classList.remove('border-gray-300');
                field.classList.add('border-red-500');
                
                if (!errorEl) {
                    const error = document.createElement('p');
                    error.className = 'field-error text-red-500 text-xs mt-1';
                    error.textContent = message;
                    field.parentElement.appendChild(error);
                } else {
                    errorEl.textContent = message;
                }
            }
        },

        /**
         * Validate entire form
         * @param {HTMLFormElement} form - Form element
         * @returns {boolean} Is valid
         */
        validateForm(form) {
            const fields = form.querySelectorAll('input, textarea, select');
            let isValid = true;

            fields.forEach(field => {
                if (!this.validateField(field)) {
                    isValid = false;
                }
            });

            return isValid;
        },
    };

    window.InsightBoost.Validation = Validation;

    // ============================================
    // Local Storage Helpers
    // ============================================
    const Storage = {
        prefix: 'insightboost_',

        get(key) {
            try {
                const item = localStorage.getItem(this.prefix + key);
                return item ? JSON.parse(item) : null;
            } catch (e) {
                return null;
            }
        },

        set(key, value) {
            try {
                localStorage.setItem(this.prefix + key, JSON.stringify(value));
                return true;
            } catch (e) {
                return false;
            }
        },

        remove(key) {
            localStorage.removeItem(this.prefix + key);
        },

        clear() {
            Object.keys(localStorage)
                .filter(key => key.startsWith(this.prefix))
                .forEach(key => localStorage.removeItem(key));
        },
    };

    window.InsightBoost.Storage = Storage;

    // ============================================
    // Console Branding
    // ============================================
    console.log(
        '%c InsightBoost %c AI-Enhanced Data Insights ',
        'background: #0284c7; color: white; padding: 5px; border-radius: 4px 0 0 4px;',
        'background: #d946ef; color: white; padding: 5px; border-radius: 0 4px 4px 0;'
    );

})();
