/**
 * InsightBoost - Visualizations Module
 * Handles chart generation, rendering, and export functionality
 */

(function() {
    'use strict';

    const Visualizations = {
        // Chart type configurations
        chartTypes: {
            scatter: {
                name: 'Scatter Plot',
                icon: 'scatter',
                description: 'Show relationships between two numeric variables',
                requiredColumns: 2,
                columnTypes: ['numeric', 'numeric'],
            },
            line: {
                name: 'Line Chart',
                icon: 'line',
                description: 'Display trends over time or continuous data',
                requiredColumns: 2,
                columnTypes: ['any', 'numeric'],
            },
            bar: {
                name: 'Bar Chart',
                icon: 'bar',
                description: 'Compare values across categories',
                requiredColumns: 2,
                columnTypes: ['categorical', 'numeric'],
            },
            histogram: {
                name: 'Histogram',
                icon: 'histogram',
                description: 'Show distribution of a single numeric variable',
                requiredColumns: 1,
                columnTypes: ['numeric'],
            },
            box: {
                name: 'Box Plot',
                icon: 'box',
                description: 'Display statistical distribution with quartiles',
                requiredColumns: 1,
                columnTypes: ['numeric'],
            },
            violin: {
                name: 'Violin Plot',
                icon: 'violin',
                description: 'Combine box plot with kernel density estimation',
                requiredColumns: 1,
                columnTypes: ['numeric'],
            },
            heatmap: {
                name: 'Heatmap',
                icon: 'heatmap',
                description: 'Show patterns in matrix data with colors',
                requiredColumns: 3,
                columnTypes: ['categorical', 'categorical', 'numeric'],
            },
            pie: {
                name: 'Pie Chart',
                icon: 'pie',
                description: 'Show proportions of a whole',
                requiredColumns: 2,
                columnTypes: ['categorical', 'numeric'],
            },
            area: {
                name: 'Area Chart',
                icon: 'area',
                description: 'Display cumulative totals over time',
                requiredColumns: 2,
                columnTypes: ['any', 'numeric'],
            },
            bubble: {
                name: 'Bubble Chart',
                icon: 'bubble',
                description: 'Scatter plot with size dimension',
                requiredColumns: 3,
                columnTypes: ['numeric', 'numeric', 'numeric'],
            },
        },

        // Default Plotly layout
        defaultLayout: {
            autosize: true,
            margin: { t: 40, r: 20, b: 40, l: 50 },
            font: { family: 'Inter, sans-serif', size: 12 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
                gridcolor: '#e5e7eb',
                linecolor: '#e5e7eb',
            },
            yaxis: {
                gridcolor: '#e5e7eb',
                linecolor: '#e5e7eb',
            },
        },

        // Color palette
        colorPalette: [
            '#0ea5e9', // Primary blue
            '#10b981', // Green
            '#8b5cf6', // Purple
            '#f59e0b', // Amber
            '#ef4444', // Red
            '#ec4899', // Pink
            '#06b6d4', // Cyan
            '#84cc16', // Lime
            '#f97316', // Orange
            '#6366f1', // Indigo
        ],

        /**
         * Render a visualization in a container
         * @param {string} containerId - Container element ID
         * @param {object} figureJson - Plotly figure JSON
         * @param {object} options - Render options
         */
        render(containerId, figureJson, options = {}) {
            const container = document.getElementById(containerId);
            if (!container) {
                console.error(`Container ${containerId} not found`);
                return;
            }

            const layout = {
                ...this.defaultLayout,
                ...figureJson.layout,
                ...options.layout,
            };

            const config = {
                responsive: true,
                displayModeBar: options.showModeBar !== false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                displaylogo: false,
                ...options.config,
            };

            // Apply color palette if not specified
            if (figureJson.data) {
                figureJson.data.forEach((trace, i) => {
                    if (!trace.marker?.color && !trace.line?.color) {
                        if (trace.marker) {
                            trace.marker.color = this.colorPalette[i % this.colorPalette.length];
                        }
                        if (trace.line) {
                            trace.line.color = this.colorPalette[i % this.colorPalette.length];
                        }
                    }
                });
            }

            Plotly.newPlot(containerId, figureJson.data, layout, config);

            // Emit event
            if (window.InsightBoost?.EventBus) {
                window.InsightBoost.EventBus.emit('visualization:rendered', { containerId, figureJson });
            }
        },

        /**
         * Update an existing visualization
         * @param {string} containerId - Container element ID
         * @param {object} updates - Data and layout updates
         */
        update(containerId, updates) {
            const container = document.getElementById(containerId);
            if (!container) return;

            if (updates.data) {
                Plotly.restyle(containerId, updates.data);
            }
            if (updates.layout) {
                Plotly.relayout(containerId, updates.layout);
            }
        },

        /**
         * Resize a visualization
         * @param {string} containerId - Container element ID
         */
        resize(containerId) {
            const container = document.getElementById(containerId);
            if (!container) return;
            Plotly.Plots.resize(container);
        },

        /**
         * Export visualization to various formats
         * @param {string} containerId - Container element ID
         * @param {string} format - Export format (png, svg, json, html)
         * @param {string} filename - File name without extension
         */
        async export(containerId, format, filename = 'visualization') {
            const container = document.getElementById(containerId);
            if (!container) {
                window.showToast?.('Visualization not found', 'error');
                return;
            }

            try {
                switch (format) {
                    case 'png':
                        const pngData = await Plotly.toImage(container, {
                            format: 'png',
                            width: 1200,
                            height: 800,
                            scale: 2,
                        });
                        this.downloadDataUrl(pngData, `${filename}.png`);
                        break;

                    case 'svg':
                        const svgData = await Plotly.toImage(container, {
                            format: 'svg',
                            width: 1200,
                            height: 800,
                        });
                        this.downloadDataUrl(svgData, `${filename}.svg`);
                        break;

                    case 'json':
                        const jsonData = {
                            data: container.data,
                            layout: container.layout,
                        };
                        window.InsightBoost?.Utils?.downloadJSON(jsonData, `${filename}.json`);
                        break;

                    case 'html':
                        const htmlContent = this.generateStandaloneHtml(container);
                        const blob = new Blob([htmlContent], { type: 'text/html' });
                        window.InsightBoost?.Utils?.downloadBlob(blob, `${filename}.html`);
                        break;

                    default:
                        throw new Error(`Unsupported format: ${format}`);
                }

                window.showToast?.(`Exported as ${format.toUpperCase()}`, 'success');
            } catch (error) {
                console.error('Export failed:', error);
                window.showToast?.('Export failed', 'error');
            }
        },

        /**
         * Download a data URL as a file
         * @param {string} dataUrl - Data URL
         * @param {string} filename - File name
         */
        downloadDataUrl(dataUrl, filename) {
            const link = document.createElement('a');
            link.href = dataUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        },

        /**
         * Generate standalone HTML for a visualization
         * @param {HTMLElement} container - Plotly container element
         * @returns {string} HTML content
         */
        generateStandaloneHtml(container) {
            const data = JSON.stringify(container.data);
            const layout = JSON.stringify(container.layout);

            return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InsightBoost Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { margin: 0; padding: 20px; font-family: Inter, sans-serif; }
        #chart { width: 100%; height: 90vh; }
        .header { margin-bottom: 20px; }
        .header h1 { margin: 0; color: #0284c7; }
        .header p { color: #6b7280; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>InsightBoost Visualization</h1>
        <p>Generated on ${new Date().toLocaleString()}</p>
    </div>
    <div id="chart"></div>
    <script>
        const data = ${data};
        const layout = ${layout};
        layout.autosize = true;
        Plotly.newPlot('chart', data, layout, {responsive: true});
    </script>
</body>
</html>`;
        },

        /**
         * Create a quick chart from data
         * @param {string} containerId - Container element ID
         * @param {string} chartType - Chart type
         * @param {object} data - Chart data
         * @param {object} options - Chart options
         */
        quickChart(containerId, chartType, data, options = {}) {
            let traces = [];
            let layout = { ...this.defaultLayout, ...options.layout };

            switch (chartType) {
                case 'scatter':
                    traces.push({
                        type: 'scatter',
                        mode: 'markers',
                        x: data.x,
                        y: data.y,
                        marker: { size: 8, color: this.colorPalette[0] },
                        name: options.name || 'Data',
                    });
                    break;

                case 'line':
                    traces.push({
                        type: 'scatter',
                        mode: 'lines+markers',
                        x: data.x,
                        y: data.y,
                        line: { width: 2, color: this.colorPalette[0] },
                        marker: { size: 6 },
                        name: options.name || 'Data',
                    });
                    break;

                case 'bar':
                    traces.push({
                        type: 'bar',
                        x: data.x,
                        y: data.y,
                        marker: { color: this.colorPalette[0] },
                        name: options.name || 'Data',
                    });
                    break;

                case 'histogram':
                    traces.push({
                        type: 'histogram',
                        x: data.x,
                        marker: { color: this.colorPalette[0] },
                        nbinsx: options.bins || 20,
                        name: options.name || 'Distribution',
                    });
                    break;

                case 'box':
                    traces.push({
                        type: 'box',
                        y: data.y,
                        marker: { color: this.colorPalette[0] },
                        name: options.name || 'Data',
                    });
                    break;

                case 'pie':
                    traces.push({
                        type: 'pie',
                        labels: data.labels,
                        values: data.values,
                        marker: { colors: this.colorPalette },
                        textinfo: 'percent+label',
                    });
                    layout.showlegend = true;
                    break;

                case 'heatmap':
                    traces.push({
                        type: 'heatmap',
                        x: data.x,
                        y: data.y,
                        z: data.z,
                        colorscale: 'Blues',
                    });
                    break;

                default:
                    console.error(`Unknown chart type: ${chartType}`);
                    return;
            }

            // Add title if provided
            if (options.title) {
                layout.title = { text: options.title, font: { size: 16 } };
            }

            // Add axis labels if provided
            if (options.xLabel) {
                layout.xaxis = { ...layout.xaxis, title: options.xLabel };
            }
            if (options.yLabel) {
                layout.yaxis = { ...layout.yaxis, title: options.yLabel };
            }

            this.render(containerId, { data: traces, layout });
        },

        /**
         * Create correlation heatmap
         * @param {string} containerId - Container element ID
         * @param {object} correlationMatrix - Correlation matrix data
         */
        correlationHeatmap(containerId, correlationMatrix) {
            const { columns, values } = correlationMatrix;

            const trace = {
                type: 'heatmap',
                x: columns,
                y: columns,
                z: values,
                colorscale: [
                    [0, '#ef4444'],    // -1: red
                    [0.5, '#ffffff'],  // 0: white
                    [1, '#0ea5e9'],    // 1: blue
                ],
                zmin: -1,
                zmax: 1,
                colorbar: { title: 'Correlation' },
            };

            const layout = {
                ...this.defaultLayout,
                title: 'Correlation Matrix',
                xaxis: { tickangle: -45 },
            };

            this.render(containerId, { data: [trace], layout });
        },

        /**
         * Create distribution comparison
         * @param {string} containerId - Container element ID
         * @param {Array} distributions - Array of {name, values} objects
         */
        distributionComparison(containerId, distributions) {
            const traces = distributions.map((dist, i) => ({
                type: 'violin',
                y: dist.values,
                name: dist.name,
                box: { visible: true },
                meanline: { visible: true },
                fillcolor: this.colorPalette[i % this.colorPalette.length],
                line: { color: this.colorPalette[i % this.colorPalette.length] },
            }));

            const layout = {
                ...this.defaultLayout,
                title: 'Distribution Comparison',
                showlegend: true,
            };

            this.render(containerId, { data: traces, layout });
        },

        /**
         * Create time series chart
         * @param {string} containerId - Container element ID
         * @param {Array} series - Array of {name, x, y} objects
         * @param {object} options - Chart options
         */
        timeSeries(containerId, series, options = {}) {
            const traces = series.map((s, i) => ({
                type: 'scatter',
                mode: 'lines',
                x: s.x,
                y: s.y,
                name: s.name,
                line: { width: 2, color: this.colorPalette[i % this.colorPalette.length] },
            }));

            const layout = {
                ...this.defaultLayout,
                title: options.title || 'Time Series',
                xaxis: {
                    ...this.defaultLayout.xaxis,
                    type: 'date',
                    title: options.xLabel || 'Date',
                },
                yaxis: {
                    ...this.defaultLayout.yaxis,
                    title: options.yLabel || 'Value',
                },
                showlegend: series.length > 1,
                hovermode: 'x unified',
            };

            this.render(containerId, { data: traces, layout });
        },

        /**
         * Destroy a visualization
         * @param {string} containerId - Container element ID
         */
        destroy(containerId) {
            const container = document.getElementById(containerId);
            if (container) {
                Plotly.purge(container);
            }
        },
    };

    // Expose globally
    window.InsightBoost = window.InsightBoost || {};
    window.InsightBoost.Visualizations = Visualizations;

    // Convenience function
    window.renderVisualization = Visualizations.render.bind(Visualizations);
    window.exportVisualization = Visualizations.export.bind(Visualizations);

})();
