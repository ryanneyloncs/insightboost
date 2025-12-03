/**
 * InsightBoost - Collaboration Module
 * Handles real-time collaboration features via WebSocket
 */

(function() {
    'use strict';

    const Collaboration = {
        // Socket.IO connection
        socket: null,
        
        // Current session state
        session: null,
        userId: null,
        
        // Participants state
        participants: new Map(),
        
        // Cursor tracking
        cursors: new Map(),
        cursorColors: [
            '#3b82f6', // Blue
            '#10b981', // Green
            '#8b5cf6', // Purple
            '#ec4899', // Pink
            '#f59e0b', // Yellow
            '#ef4444', // Red
            '#06b6d4', // Cyan
            '#84cc16', // Lime
        ],
        
        // Event callbacks
        callbacks: {
            onConnect: null,
            onDisconnect: null,
            onUserJoined: null,
            onUserLeft: null,
            onCursorMove: null,
            onInsightShared: null,
            onVisualizationShared: null,
            onCommentAdded: null,
            onSessionEnded: null,
        },

        /**
         * Initialize collaboration module
         * @param {object} options - Configuration options
         */
        init(options = {}) {
            this.userId = options.userId || this.generateUserId();
            
            if (options.callbacks) {
                Object.assign(this.callbacks, options.callbacks);
            }

            // Store userId for persistence
            if (window.InsightBoost?.Storage) {
                window.InsightBoost.Storage.set('userId', this.userId);
            }

            console.log('[Collaboration] Initialized with userId:', this.userId);
        },

        /**
         * Connect to WebSocket server
         * @param {string} url - Socket.IO server URL (optional)
         */
        connect(url = null) {
            if (this.socket?.connected) {
                console.log('[Collaboration] Already connected');
                return;
            }

            try {
                this.socket = io(url || window.location.origin, {
                    transports: ['websocket', 'polling'],
                    reconnection: true,
                    reconnectionAttempts: 5,
                    reconnectionDelay: 1000,
                });

                this.setupSocketListeners();
                console.log('[Collaboration] Connecting to server...');
            } catch (error) {
                console.error('[Collaboration] Connection failed:', error);
                window.showToast?.('Failed to connect to collaboration server', 'error');
            }
        },

        /**
         * Setup socket event listeners
         */
        setupSocketListeners() {
            const socket = this.socket;

            // Connection events
            socket.on('connect', () => {
                console.log('[Collaboration] Connected');
                this.callbacks.onConnect?.();
                
                // Rejoin session if we have one
                if (this.session) {
                    this.joinSession(this.session.id);
                }
            });

            socket.on('disconnect', (reason) => {
                console.log('[Collaboration] Disconnected:', reason);
                this.callbacks.onDisconnect?.(reason);
            });

            socket.on('connect_error', (error) => {
                console.error('[Collaboration] Connection error:', error);
            });

            // Session events
            socket.on('session.user_joined', (data) => {
                console.log('[Collaboration] User joined:', data);
                this.handleUserJoined(data);
            });

            socket.on('session.user_left', (data) => {
                console.log('[Collaboration] User left:', data);
                this.handleUserLeft(data);
            });

            socket.on('session.ended', (data) => {
                console.log('[Collaboration] Session ended:', data);
                this.handleSessionEnded(data);
            });

            // Cursor events
            socket.on('cursor.moved', (data) => {
                this.handleCursorMove(data);
            });

            // Content sharing events
            socket.on('insight.created', (data) => {
                console.log('[Collaboration] Insight shared:', data);
                this.callbacks.onInsightShared?.(data);
            });

            socket.on('visualization.shared', (data) => {
                console.log('[Collaboration] Visualization shared:', data);
                this.callbacks.onVisualizationShared?.(data);
            });

            socket.on('insight.commented', (data) => {
                console.log('[Collaboration] Comment added:', data);
                this.callbacks.onCommentAdded?.(data);
            });

            // Error handling
            socket.on('error', (error) => {
                console.error('[Collaboration] Socket error:', error);
                window.showToast?.('Collaboration error occurred', 'error');
            });
        },

        /**
         * Create a new collaboration session
         * @param {string} datasetId - Dataset ID
         * @param {string} name - Session name
         * @param {object} options - Additional options
         * @returns {Promise<object>} Created session
         */
        async createSession(datasetId, name, options = {}) {
            try {
                const response = await window.InsightBoost.API.post('/sessions', {
                    dataset_id: datasetId,
                    name: name,
                    user_id: this.userId,
                    max_participants: options.maxParticipants || 10,
                    expires_in_hours: options.expiresIn || 24,
                });

                if (response.success) {
                    this.session = response.session;
                    this.connect();
                    this.joinSession(this.session.id);
                    return this.session;
                }
            } catch (error) {
                console.error('[Collaboration] Failed to create session:', error);
                throw error;
            }
        },

        /**
         * Join an existing session
         * @param {string} sessionId - Session ID
         */
        joinSession(sessionId) {
            if (!this.socket?.connected) {
                this.connect();
                // Will rejoin after connection
                return;
            }

            this.socket.emit('join_session', {
                session_id: sessionId,
                user_id: this.userId,
            });

            console.log('[Collaboration] Joining session:', sessionId);
        },

        /**
         * Leave current session
         */
        leaveSession() {
            if (!this.socket || !this.session) return;

            this.socket.emit('leave_session', {
                session_id: this.session.id,
                user_id: this.userId,
            });

            this.cleanup();
            console.log('[Collaboration] Left session');
        },

        /**
         * Handle user joined event
         * @param {object} data - Event data
         */
        handleUserJoined(data) {
            if (data.user_id === this.userId) return;

            this.participants.set(data.user_id, {
                userId: data.user_id,
                joinedAt: new Date(),
                color: this.getColorForUser(data.user_id),
            });

            this.callbacks.onUserJoined?.(data);
            window.showToast?.(`${this.getDisplayName(data.user_id)} joined the session`, 'info');
        },

        /**
         * Handle user left event
         * @param {object} data - Event data
         */
        handleUserLeft(data) {
            this.participants.delete(data.user_id);
            this.removeCursor(data.user_id);
            
            this.callbacks.onUserLeft?.(data);
            window.showToast?.(`${this.getDisplayName(data.user_id)} left the session`, 'info');
        },

        /**
         * Handle session ended event
         * @param {object} data - Event data
         */
        handleSessionEnded(data) {
            this.callbacks.onSessionEnded?.(data);
            this.cleanup();
            window.showToast?.('Session has ended', 'warning');
        },

        /**
         * Handle cursor move event
         * @param {object} data - Cursor data
         */
        handleCursorMove(data) {
            if (data.user_id === this.userId) return;

            this.updateCursor(data);
            this.callbacks.onCursorMove?.(data);
        },

        /**
         * Send cursor position update
         * @param {number} x - X position (0-1)
         * @param {number} y - Y position (0-1)
         * @param {string} elementId - Optional element ID being hovered
         */
        sendCursorPosition(x, y, elementId = null) {
            if (!this.socket?.connected || !this.session) return;

            this.socket.emit('cursor_move', {
                session_id: this.session.id,
                user_id: this.userId,
                x: x,
                y: y,
                element_id: elementId,
            });
        },

        /**
         * Share an insight with the session
         * @param {object} insight - Insight data
         */
        shareInsight(insight) {
            if (!this.socket?.connected || !this.session) return;

            this.socket.emit('share_insight', {
                session_id: this.session.id,
                user_id: this.userId,
                insight: insight,
            });
        },

        /**
         * Share a visualization with the session
         * @param {object} visualization - Visualization data
         */
        shareVisualization(visualization) {
            if (!this.socket?.connected || !this.session) return;

            this.socket.emit('share_visualization', {
                session_id: this.session.id,
                user_id: this.userId,
                visualization: visualization,
            });
        },

        /**
         * Add a comment
         * @param {string} content - Comment content
         * @param {object} options - Comment options (insightId, visualizationId, parentId)
         */
        addComment(content, options = {}) {
            if (!this.socket?.connected || !this.session) return;

            this.socket.emit('add_comment', {
                session_id: this.session.id,
                user_id: this.userId,
                content: content,
                insight_id: options.insightId,
                visualization_id: options.visualizationId,
                parent_id: options.parentId,
            });
        },

        /**
         * Update remote cursor on screen
         * @param {object} data - Cursor data
         */
        updateCursor(data) {
            const cursorId = `cursor-${data.user_id}`;
            let cursor = document.getElementById(cursorId);
            const overlay = document.getElementById('cursor-overlay');

            if (!overlay) return;

            if (!cursor) {
                cursor = this.createCursorElement(data.user_id);
                overlay.appendChild(cursor);
            }

            cursor.style.left = `${data.x * 100}%`;
            cursor.style.top = `${data.y * 100}%`;
            cursor.style.opacity = '1';

            // Store cursor data
            this.cursors.set(data.user_id, {
                ...data,
                lastUpdate: Date.now(),
            });

            // Hide cursor after inactivity
            clearTimeout(cursor._hideTimeout);
            cursor._hideTimeout = setTimeout(() => {
                cursor.style.opacity = '0.3';
            }, 3000);
        },

        /**
         * Create cursor DOM element
         * @param {string} userId - User ID
         * @returns {HTMLElement} Cursor element
         */
        createCursorElement(userId) {
            const color = this.getColorForUser(userId);
            const displayName = this.getDisplayName(userId);

            const cursor = document.createElement('div');
            cursor.id = `cursor-${userId}`;
            cursor.className = 'remote-cursor';
            cursor.innerHTML = `
                <svg class="remote-cursor-pointer" viewBox="0 0 24 24" fill="${color}">
                    <path d="M4 4l16 8-8 3-3 8-5-19z"/>
                </svg>
                <span class="remote-cursor-label" style="background-color: ${color}">
                    ${displayName}
                </span>
            `;

            return cursor;
        },

        /**
         * Remove cursor from screen
         * @param {string} userId - User ID
         */
        removeCursor(userId) {
            const cursor = document.getElementById(`cursor-${userId}`);
            if (cursor) {
                cursor.remove();
            }
            this.cursors.delete(userId);
        },

        /**
         * Setup cursor tracking on an element
         * @param {HTMLElement} element - Element to track
         */
        setupCursorTracking(element) {
            if (!element) return;

            const throttledMove = window.InsightBoost?.Utils?.throttle((e) => {
                const rect = element.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width;
                const y = (e.clientY - rect.top) / rect.height;
                this.sendCursorPosition(x, y);
            }, 50);

            element.addEventListener('mousemove', throttledMove);
            element.addEventListener('mouseleave', () => {
                this.sendCursorPosition(-1, -1); // Off-screen indicator
            });
        },

        /**
         * Get color for a user
         * @param {string} userId - User ID
         * @returns {string} Color hex code
         */
        getColorForUser(userId) {
            // Generate consistent color based on userId
            let hash = 0;
            for (let i = 0; i < userId.length; i++) {
                hash = userId.charCodeAt(i) + ((hash << 5) - hash);
            }
            return this.cursorColors[Math.abs(hash) % this.cursorColors.length];
        },

        /**
         * Get display name for user
         * @param {string} userId - User ID
         * @returns {string} Display name
         */
        getDisplayName(userId) {
            if (userId === this.userId) return 'You';
            // Extract short ID for display
            return userId.replace('user-', '').substring(0, 6);
        },

        /**
         * Generate a random user ID
         * @returns {string} User ID
         */
        generateUserId() {
            const stored = window.InsightBoost?.Storage?.get('userId');
            if (stored) return stored;
            return 'user-' + Math.random().toString(36).substr(2, 9);
        },

        /**
         * Get connection status
         * @returns {string} Status string
         */
        getStatus() {
            if (!this.socket) return 'disconnected';
            return this.socket.connected ? 'connected' : 'connecting';
        },

        /**
         * Get current session info
         * @returns {object|null} Session info
         */
        getSession() {
            return this.session;
        },

        /**
         * Get list of participants
         * @returns {Array} Participants array
         */
        getParticipants() {
            return Array.from(this.participants.values());
        },

        /**
         * Check if user is session owner
         * @returns {boolean}
         */
        isOwner() {
            return this.session?.owner_id === this.userId;
        },

        /**
         * End the session (owner only)
         */
        async endSession() {
            if (!this.session || !this.isOwner()) return;

            try {
                await window.InsightBoost.API.delete(`/sessions/${this.session.id}`);
                this.cleanup();
            } catch (error) {
                console.error('[Collaboration] Failed to end session:', error);
                throw error;
            }
        },

        /**
         * Cleanup resources
         */
        cleanup() {
            // Clear cursors
            this.cursors.forEach((_, userId) => this.removeCursor(userId));
            this.cursors.clear();

            // Clear participants
            this.participants.clear();

            // Clear session
            this.session = null;
        },

        /**
         * Disconnect from server
         */
        disconnect() {
            if (this.socket) {
                this.socket.disconnect();
                this.socket = null;
            }
            this.cleanup();
        },
    };

    // ============================================
    // Presence Indicator Component
    // ============================================
    const PresenceIndicator = {
        container: null,

        /**
         * Initialize presence indicator
         * @param {string} containerId - Container element ID
         */
        init(containerId) {
            this.container = document.getElementById(containerId);
            if (!this.container) return;

            this.render();

            // Update on participant changes
            if (window.InsightBoost?.EventBus) {
                window.InsightBoost.EventBus.on('collaboration:participant_change', () => {
                    this.render();
                });
            }
        },

        /**
         * Render presence indicator
         */
        render() {
            if (!this.container) return;

            const participants = Collaboration.getParticipants();
            const maxVisible = 4;

            let html = '<div class="flex -space-x-2">';

            // Render visible avatars
            participants.slice(0, maxVisible).forEach(p => {
                html += `
                    <div class="h-8 w-8 rounded-full flex items-center justify-center text-white text-xs font-medium border-2 border-white"
                         style="background-color: ${p.color}"
                         title="${Collaboration.getDisplayName(p.userId)}">
                        ${Collaboration.getDisplayName(p.userId).substring(0, 2).toUpperCase()}
                    </div>
                `;
            });

            // Render overflow indicator
            if (participants.length > maxVisible) {
                html += `
                    <div class="h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-600 text-xs font-medium border-2 border-white">
                        +${participants.length - maxVisible}
                    </div>
                `;
            }

            html += '</div>';

            // Add status indicator
            const status = Collaboration.getStatus();
            const statusColor = status === 'connected' ? 'bg-green-400' : 'bg-yellow-400';
            html += `
                <div class="flex items-center ml-3">
                    <span class="h-2 w-2 ${statusColor} rounded-full ${status === 'connecting' ? 'animate-pulse' : ''}"></span>
                    <span class="ml-2 text-sm text-gray-500 capitalize">${status}</span>
                </div>
            `;

            this.container.innerHTML = html;
        },
    };

    // Expose globally
    window.InsightBoost = window.InsightBoost || {};
    window.InsightBoost.Collaboration = Collaboration;
    window.InsightBoost.PresenceIndicator = PresenceIndicator;

})();
