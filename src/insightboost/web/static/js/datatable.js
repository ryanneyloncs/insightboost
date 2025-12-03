/**
 * InsightBoost - Data Table Module
 * Handles paginated, sortable data table functionality
 */

(function() {
    'use strict';

    class DataTable {
        /**
         * Create a DataTable instance
         * @param {string} containerId - Container element ID
         * @param {object} options - Configuration options
         */
        constructor(containerId, options = {}) {
            this.container = document.getElementById(containerId);
            if (!this.container) {
                throw new Error(`Container ${containerId} not found`);
            }

            // Configuration
            this.options = {
                pageSize: options.pageSize || 50,
                maxPageSize: options.maxPageSize || 1000,
                sortable: options.sortable !== false,
                selectable: options.selectable || false,
                onRowClick: options.onRowClick || null,
                onSort: options.onSort || null,
                onPageChange: options.onPageChange || null,
                formatters: options.formatters || {},
                emptyMessage: options.emptyMessage || 'No data available',
                loadingMessage: options.loadingMessage || 'Loading data...',
            };

            // State
            this.data = [];
            this.columns = [];
            this.currentPage = 1;
            this.totalPages = 1;
            this.totalRows = 0;
            this.sortColumn = null;
            this.sortDirection = 'asc';
            this.selectedRows = new Set();
            this.isLoading = false;

            // Build table structure
            this.buildStructure();
        }

        /**
         * Build table HTML structure
         */
        buildStructure() {
            this.container.innerHTML = `
                <div class="data-table-wrapper">
                    <div class="data-table-toolbar">
                        <div class="data-table-info">
                            <span class="data-table-count"></span>
                        </div>
                        <div class="data-table-actions">
                            <select class="data-table-page-size">
                                <option value="25">25 rows</option>
                                <option value="50" selected>50 rows</option>
                                <option value="100">100 rows</option>
                                <option value="250">250 rows</option>
                            </select>
                        </div>
                    </div>
                    <div class="data-table-scroll">
                        <table class="data-table">
                            <thead></thead>
                            <tbody></tbody>
                        </table>
                    </div>
                    <div class="data-table-footer">
                        <div class="data-table-pagination">
                            <button class="btn-prev" disabled>
                                <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                                </svg>
                                Previous
                            </button>
                            <span class="page-info"></span>
                            <button class="btn-next" disabled>
                                Next
                                <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            `;

            // Cache DOM references
            this.thead = this.container.querySelector('thead');
            this.tbody = this.container.querySelector('tbody');
            this.countEl = this.container.querySelector('.data-table-count');
            this.pageInfoEl = this.container.querySelector('.page-info');
            this.prevBtn = this.container.querySelector('.btn-prev');
            this.nextBtn = this.container.querySelector('.btn-next');
            this.pageSizeSelect = this.container.querySelector('.data-table-page-size');

            // Bind events
            this.bindEvents();
        }

        /**
         * Bind event listeners
         */
        bindEvents() {
            // Pagination
            this.prevBtn.addEventListener('click', () => this.prevPage());
            this.nextBtn.addEventListener('click', () => this.nextPage());

            // Page size
            this.pageSizeSelect.addEventListener('change', (e) => {
                this.options.pageSize = parseInt(e.target.value);
                this.currentPage = 1;
                this.render();
                this.options.onPageChange?.(this.currentPage, this.options.pageSize);
            });

            // Row click
            this.tbody.addEventListener('click', (e) => {
                const row = e.target.closest('tr');
                if (row && this.options.onRowClick) {
                    const rowIndex = parseInt(row.dataset.index);
                    this.options.onRowClick(this.data[rowIndex], rowIndex);
                }
            });
        }

        /**
         * Set columns configuration
         * @param {Array} columns - Column definitions
         */
        setColumns(columns) {
            this.columns = columns.map(col => {
                if (typeof col === 'string') {
                    return { key: col, label: col, sortable: true };
                }
                return {
                    key: col.key || col.name,
                    label: col.label || col.name || col.key,
                    sortable: col.sortable !== false,
                    width: col.width,
                    align: col.align || 'left',
                    formatter: col.formatter || this.options.formatters[col.key],
                };
            });
            this.renderHeader();
        }

        /**
         * Set data
         * @param {Array} data - Data rows
         * @param {object} pagination - Pagination info
         */
        setData(data, pagination = null) {
            this.data = data || [];
            
            if (pagination) {
                this.currentPage = pagination.page || 1;
                this.totalPages = pagination.total_pages || 1;
                this.totalRows = pagination.total_rows || data.length;
            } else {
                this.totalRows = this.data.length;
                this.totalPages = Math.ceil(this.totalRows / this.options.pageSize);
            }

            this.render();
        }

        /**
         * Render table header
         */
        renderHeader() {
            let html = '<tr>';

            if (this.options.selectable) {
                html += `
                    <th class="w-10">
                        <input type="checkbox" class="select-all" />
                    </th>
                `;
            }

            this.columns.forEach(col => {
                const sortable = this.options.sortable && col.sortable;
                const isSorted = this.sortColumn === col.key;
                const sortIcon = isSorted
                    ? this.sortDirection === 'asc'
                        ? '↑'
                        : '↓'
                    : '';

                html += `
                    <th class="${sortable ? 'sortable' : ''} ${isSorted ? 'sorted' : ''}"
                        data-column="${col.key}"
                        style="${col.width ? `width: ${col.width}` : ''}; text-align: ${col.align}">
                        ${col.label}
                        ${sortable ? `<span class="sort-icon">${sortIcon}</span>` : ''}
                    </th>
                `;
            });

            html += '</tr>';
            this.thead.innerHTML = html;

            // Bind sort events
            if (this.options.sortable) {
                this.thead.querySelectorAll('th.sortable').forEach(th => {
                    th.addEventListener('click', () => this.sort(th.dataset.column));
                });
            }

            // Bind select all
            if (this.options.selectable) {
                this.thead.querySelector('.select-all')?.addEventListener('change', (e) => {
                    this.toggleSelectAll(e.target.checked);
                });
            }
        }

        /**
         * Render table body
         */
        render() {
            if (this.isLoading) {
                this.tbody.innerHTML = `
                    <tr>
                        <td colspan="${this.columns.length + (this.options.selectable ? 1 : 0)}" class="text-center py-8">
                            <div class="spinner mx-auto mb-2"></div>
                            ${this.options.loadingMessage}
                        </td>
                    </tr>
                `;
                return;
            }

            if (this.data.length === 0) {
                this.tbody.innerHTML = `
                    <tr>
                        <td colspan="${this.columns.length + (this.options.selectable ? 1 : 0)}" class="text-center py-8 text-gray-500">
                            ${this.options.emptyMessage}
                        </td>
                    </tr>
                `;
                this.updatePagination();
                return;
            }

            let html = '';
            this.data.forEach((row, index) => {
                const isSelected = this.selectedRows.has(index);
                html += `<tr data-index="${index}" class="${isSelected ? 'selected' : ''} ${this.options.onRowClick ? 'cursor-pointer' : ''}">`;

                if (this.options.selectable) {
                    html += `
                        <td class="w-10">
                            <input type="checkbox" class="row-select" data-index="${index}" ${isSelected ? 'checked' : ''} />
                        </td>
                    `;
                }

                this.columns.forEach(col => {
                    let value = row[col.key];
                    
                    // Apply formatter if exists
                    if (col.formatter) {
                        value = col.formatter(value, row, index);
                    } else if (value === null || value === undefined) {
                        value = '<span class="text-gray-400">—</span>';
                    } else {
                        value = this.escapeHtml(String(value));
                    }

                    html += `<td style="text-align: ${col.align}">${value}</td>`;
                });

                html += '</tr>';
            });

            this.tbody.innerHTML = html;

            // Bind row selection
            if (this.options.selectable) {
                this.tbody.querySelectorAll('.row-select').forEach(checkbox => {
                    checkbox.addEventListener('change', (e) => {
                        e.stopPropagation();
                        this.toggleRowSelection(parseInt(e.target.dataset.index), e.target.checked);
                    });
                });
            }

            this.updatePagination();
        }

        /**
         * Update pagination controls
         */
        updatePagination() {
            const start = this.data.length > 0 ? (this.currentPage - 1) * this.options.pageSize + 1 : 0;
            const end = Math.min(this.currentPage * this.options.pageSize, this.totalRows);

            this.countEl.textContent = `Showing ${start.toLocaleString()}-${end.toLocaleString()} of ${this.totalRows.toLocaleString()} rows`;
            this.pageInfoEl.textContent = `Page ${this.currentPage} of ${this.totalPages}`;

            this.prevBtn.disabled = this.currentPage <= 1;
            this.nextBtn.disabled = this.currentPage >= this.totalPages;
        }

        /**
         * Sort by column
         * @param {string} column - Column key
         */
        sort(column) {
            if (this.sortColumn === column) {
                this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                this.sortColumn = column;
                this.sortDirection = 'asc';
            }

            // If external sorting handler provided
            if (this.options.onSort) {
                this.options.onSort(this.sortColumn, this.sortDirection);
            } else {
                // Client-side sorting
                this.data.sort((a, b) => {
                    const aVal = a[column];
                    const bVal = b[column];

                    if (aVal === null || aVal === undefined) return 1;
                    if (bVal === null || bVal === undefined) return -1;

                    let comparison = 0;
                    if (typeof aVal === 'number' && typeof bVal === 'number') {
                        comparison = aVal - bVal;
                    } else {
                        comparison = String(aVal).localeCompare(String(bVal));
                    }

                    return this.sortDirection === 'asc' ? comparison : -comparison;
                });

                this.render();
            }

            this.renderHeader();
        }

        /**
         * Go to previous page
         */
        prevPage() {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.options.onPageChange?.(this.currentPage, this.options.pageSize);
            }
        }

        /**
         * Go to next page
         */
        nextPage() {
            if (this.currentPage < this.totalPages) {
                this.currentPage++;
                this.options.onPageChange?.(this.currentPage, this.options.pageSize);
            }
        }

        /**
         * Go to specific page
         * @param {number} page - Page number
         */
        goToPage(page) {
            if (page >= 1 && page <= this.totalPages) {
                this.currentPage = page;
                this.options.onPageChange?.(this.currentPage, this.options.pageSize);
            }
        }

        /**
         * Toggle row selection
         * @param {number} index - Row index
         * @param {boolean} selected - Selection state
         */
        toggleRowSelection(index, selected) {
            if (selected) {
                this.selectedRows.add(index);
            } else {
                this.selectedRows.delete(index);
            }
            this.render();
        }

        /**
         * Toggle select all rows
         * @param {boolean} selected - Selection state
         */
        toggleSelectAll(selected) {
            if (selected) {
                this.data.forEach((_, index) => this.selectedRows.add(index));
            } else {
                this.selectedRows.clear();
            }
            this.render();
        }

        /**
         * Get selected rows
         * @returns {Array} Selected row data
         */
        getSelectedRows() {
            return Array.from(this.selectedRows).map(index => this.data[index]);
        }

        /**
         * Set loading state
         * @param {boolean} loading - Loading state
         */
        setLoading(loading) {
            this.isLoading = loading;
            this.render();
        }

        /**
         * Escape HTML to prevent XSS
         * @param {string} str - String to escape
         * @returns {string} Escaped string
         */
        escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }

        /**
         * Refresh the table
         */
        refresh() {
            this.render();
        }

        /**
         * Destroy the table
         */
        destroy() {
            this.container.innerHTML = '';
        }
    }

    // Common formatters
    DataTable.formatters = {
        number: (value) => {
            if (value === null || value === undefined) return '—';
            return Number(value).toLocaleString();
        },

        decimal: (decimals = 2) => (value) => {
            if (value === null || value === undefined) return '—';
            return Number(value).toFixed(decimals);
        },

        percent: (value) => {
            if (value === null || value === undefined) return '—';
            return (Number(value) * 100).toFixed(1) + '%';
        },

        date: (value) => {
            if (!value) return '—';
            return new Date(value).toLocaleDateString();
        },

        datetime: (value) => {
            if (!value) return '—';
            return new Date(value).toLocaleString();
        },

        boolean: (value) => {
            if (value === true) return '<span class="text-green-600">Yes</span>';
            if (value === false) return '<span class="text-red-600">No</span>';
            return '—';
        },

        truncate: (maxLength = 50) => (value) => {
            if (!value) return '—';
            const str = String(value);
            if (str.length <= maxLength) return str;
            return `<span title="${str}">${str.substring(0, maxLength)}...</span>`;
        },

        badge: (colorMap = {}) => (value) => {
            const color = colorMap[value] || 'gray';
            return `<span class="badge badge-${color}">${value}</span>`;
        },
    };

    // Expose globally
    window.InsightBoost = window.InsightBoost || {};
    window.InsightBoost.DataTable = DataTable;

})();
