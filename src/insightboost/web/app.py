"""
Flask application factory for InsightBoost.

This module provides the main Flask application with all necessary
configuration, extensions, and route registration.
"""

from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_login import LoginManager
from flask_socketio import SocketIO

from insightboost.config.logging_config import get_logger, setup_logging
from insightboost.config.settings import get_settings
from insightboost.utils.exceptions import InsightBoostError

# Initialize extensions
socketio = SocketIO()
login_manager = LoginManager()
logger = get_logger("web.app")


def create_app(config_override: dict | None = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_override: Optional configuration overrides

    Returns:
        Configured Flask application
    """
    # Set up logging
    setup_logging()

    # Get settings
    settings = get_settings()

    # Create Flask app
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    # Configure app
    app.config.update(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.max_upload_size_bytes,
        SQLALCHEMY_DATABASE_URI=settings.database_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        DEBUG=settings.flask_debug,
        ENV=settings.flask_env,
    )

    # Apply config overrides
    if config_override:
        app.config.update(config_override)

    # Initialize extensions with environment-appropriate CORS
    if settings.is_production:
        # In production, restrict to configured origins
        allowed_origins = settings.allowed_origins
        CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
        socketio.init_app(app, cors_allowed_origins=allowed_origins, async_mode="eventlet")
    else:
        # In development, allow all origins for easier testing
        CORS(app, resources={r"/api/*": {"origins": "*"}})
        socketio.init_app(app, cors_allowed_origins="*", async_mode="eventlet")
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # Register blueprints
    from insightboost.web.routes.collaboration import collaboration_bp
    from insightboost.web.routes.insights import insights_bp
    from insightboost.web.routes.visualizations import visualizations_bp

    app.register_blueprint(insights_bp, url_prefix="/api/v1")
    app.register_blueprint(visualizations_bp, url_prefix="/api/v1")
    app.register_blueprint(collaboration_bp, url_prefix="/api/v1")

    # Register error handlers
    register_error_handlers(app)

    # Register template routes
    register_template_routes(app)

    # Register WebSocket events
    register_socketio_events(socketio)

    logger.info(f"InsightBoost app created (env={settings.flask_env})")

    return app


def register_error_handlers(app: Flask) -> None:
    """Register error handlers for the application."""

    @app.errorhandler(InsightBoostError)
    def handle_insightboost_error(error: InsightBoostError):
        """Handle custom InsightBoost errors."""
        logger.error(f"InsightBoostError: {error.message}")
        response = jsonify(error.to_dict())
        response.status_code = 400
        return response

    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle bad request errors."""
        # Don't expose internal error details in production
        from insightboost.config.settings import get_settings
        _settings = get_settings()
        details = str(error) if not _settings.is_production else {}
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "BAD_REQUEST",
                    "message": "Bad request",
                    "details": details,
                }
            ),
            400,
        )

    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle not found errors."""
        from insightboost.config.settings import get_settings
        _settings = get_settings()
        details = str(error) if not _settings.is_production else {}
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Resource not found",
                    "details": details,
                }
            ),
            404,
        )

    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle internal server errors."""
        logger.error(f"Internal server error: {error}")
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                    "details": {},
                }
            ),
            500,
        )

    @app.errorhandler(413)
    def handle_file_too_large(error):
        """Handle file too large errors."""
        settings = get_settings()
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "FILE_TOO_LARGE",
                    "message": f"File exceeds maximum size of {settings.max_upload_size_mb} MB",
                    "details": {},
                }
            ),
            413,
        )


def register_template_routes(app: Flask) -> None:
    """Register template-based routes for the web UI."""

    @app.route("/")
    def index():
        """Render the main dashboard."""
        return render_template("dashboard.html")

    @app.route("/analysis/<dataset_id>")
    def analysis(dataset_id: str):
        """Render the analysis view for a dataset."""
        return render_template("analysis.html", dataset_id=dataset_id)

    @app.route("/collaborate/<session_id>")
    def collaborate(session_id: str):
        """Render the collaboration view."""
        return render_template("collaborate.html", session_id=session_id)

    @app.route("/health")
    def health():
        """Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "version": "0.1.0",
            }
        )

    @app.route("/api/v1/info")
    def api_info():
        """API information endpoint."""
        return jsonify(
            {
                "name": "InsightBoost API",
                "version": "v1",
                "documentation": "/api/v1/docs",
                "endpoints": {
                    "datasets": "/api/v1/datasets",
                    "insights": "/api/v1/datasets/{id}/insights",
                    "visualizations": "/api/v1/datasets/{id}/visualizations",
                    "sessions": "/api/v1/sessions",
                },
            }
        )


def register_socketio_events(sio: SocketIO) -> None:
    """Register WebSocket event handlers for real-time collaboration."""

    @sio.on("connect")
    def handle_connect():
        """Handle client connection."""
        logger.debug(f"Client connected: {request.sid}")

    @sio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection."""
        logger.debug(f"Client disconnected: {request.sid}")

    @sio.on("join_session")
    def handle_join_session(data: dict):
        """Handle user joining a collaboration session."""
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        if not session_id or not user_id:
            from flask_socketio import emit
            emit("error", {"message": "session_id and user_id are required"})
            return

        # Import session storage to verify session exists and user can join
        from insightboost.web.routes.collaboration import _sessions
        
        if session_id not in _sessions:
            from flask_socketio import emit
            emit("error", {"message": "Session not found"})
            return
            
        session = _sessions[session_id]
        
        # Check if session is active
        if session.get("status") != "active":
            from flask_socketio import emit
            emit("error", {"message": "Session is not active"})
            return
            
        # Check participant limit
        if len(session.get("participants", [])) >= session.get("max_participants", 10):
            if user_id not in session.get("participants", []):
                from flask_socketio import emit
                emit("error", {"message": "Session is full"})
                return

        from flask_socketio import emit, join_room

        join_room(session_id)
        emit(
            "session.user_joined",
            {
                "user_id": user_id,
                "session_id": session_id,
            },
            to=session_id,
        )
        logger.info(f"User {user_id} joined session {session_id}")

    @sio.on("leave_session")
    def handle_leave_session(data: dict):
        """Handle user leaving a collaboration session."""
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        if session_id and user_id:
            from flask_socketio import emit, leave_room

            leave_room(session_id)
            emit(
                "session.user_left",
                {
                    "user_id": user_id,
                    "session_id": session_id,
                },
                to=session_id,
            )
            logger.info(f"User {user_id} left session {session_id}")

    @sio.on("cursor_move")
    def handle_cursor_move(data: dict):
        """Handle cursor movement for shared exploration."""
        session_id = data.get("session_id")
        if session_id:
            from flask_socketio import emit

            emit("cursor.moved", data, to=session_id, include_self=False)

    @sio.on("share_insight")
    def handle_share_insight(data: dict):
        """Handle sharing an insight with the session."""
        session_id = data.get("session_id")
        if session_id:
            from flask_socketio import emit

            emit("insight.created", data, to=session_id)
            logger.info(f"Insight shared in session {session_id}")

    @sio.on("share_visualization")
    def handle_share_visualization(data: dict):
        """Handle sharing a visualization with the session."""
        session_id = data.get("session_id")
        if session_id:
            from flask_socketio import emit

            emit("visualization.shared", data, to=session_id)
            logger.info(f"Visualization shared in session {session_id}")

    @sio.on("add_comment")
    def handle_add_comment(data: dict):
        """Handle adding a comment to an insight."""
        session_id = data.get("session_id")
        if session_id:
            from flask_socketio import emit

            emit("insight.commented", data, to=session_id)


def main() -> None:
    """Run the application."""
    settings = get_settings()
    app = create_app()

    socketio.run(
        app,
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
    )


if __name__ == "__main__":
    main()
