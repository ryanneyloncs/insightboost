"""
Collaboration API routes for InsightBoost.

This module provides API endpoints for real-time collaboration sessions.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any

from flask import Blueprint, jsonify, request

from insightboost.api.rate_limiter import rate_limit
from insightboost.config.logging_config import get_logger
from insightboost.models.user import SessionStatus

logger = get_logger("routes.collaboration")
collaboration_bp = Blueprint("collaboration", __name__)

# In-memory storage for sessions (replace with database in production)
_sessions: dict[str, dict[str, Any]] = {}
_session_comments: dict[str, list[dict[str, Any]]] = {}
_session_snapshots: dict[str, list[dict[str, Any]]] = {}
_active_cursors: dict[str, dict[str, dict[str, Any]]] = (
    {}
)  # session_id -> user_id -> cursor


def get_datasets_storage() -> dict:
    """Get access to the datasets storage from insights module."""
    from insightboost.web.routes.insights import _datasets

    return _datasets


# =============================================================================
# Session Management Endpoints
# =============================================================================


@collaboration_bp.route("/sessions", methods=["POST"])
@rate_limit(
    requests_per_minute=10, error_message="Session creation rate limit exceeded."
)
def create_session():
    """
    Create a new collaboration session.

    Request body:
        {
            "dataset_id": "uuid",
            "name": "Session name",
            "max_participants": 10,
            "expires_in_hours": 24
        }
    """
    try:
        data = request.get_json() or {}

        dataset_id = data.get("dataset_id")
        if not dataset_id:
            return (
                jsonify(
                    {
                        "error": True,
                        "error_code": "MISSING_FIELD",
                        "message": "dataset_id is required",
                    }
                ),
                400,
            )

        # Verify dataset exists
        _datasets = get_datasets_storage()
        if dataset_id not in _datasets:
            return (
                jsonify(
                    {
                        "error": True,
                        "error_code": "NOT_FOUND",
                        "message": "Dataset not found",
                    }
                ),
                404,
            )

        # Get session parameters
        name = data.get("name", f"Session for {_datasets[dataset_id]['name']}")
        max_participants = data.get("max_participants", 10)
        expires_in_hours = data.get("expires_in_hours", 24)

        # Validate parameters
        max_participants = min(max(2, max_participants), 50)
        expires_in_hours = min(max(1, expires_in_hours), 168)  # Max 1 week

        # Create session
        session_id = str(uuid.uuid4())
        created_by = data.get("user_id", str(uuid.uuid4()))  # Placeholder

        session = {
            "id": session_id,
            "name": name,
            "dataset_id": dataset_id,
            "created_by": created_by,
            "created_at": datetime.utcnow().isoformat(),
            "status": SessionStatus.ACTIVE.value,
            "participants": [created_by],
            "max_participants": max_participants,
            "expires_at": (
                datetime.utcnow() + timedelta(hours=expires_in_hours)
            ).isoformat(),
            "settings": {
                "allow_comments": True,
                "allow_snapshots": True,
                "share_cursors": True,
            },
        }

        _sessions[session_id] = session
        _session_comments[session_id] = []
        _session_snapshots[session_id] = []
        _active_cursors[session_id] = {}

        logger.info(f"Session created: {session_id} for dataset {dataset_id}")

        return (
            jsonify(
                {
                    "success": True,
                    "session": session,
                }
            ),
            201,
        )

    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "CREATION_FAILED",
                    "message": str(e),
                }
            ),
            500,
        )


@collaboration_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """List all active collaboration sessions."""
    # Optional filters
    dataset_id = request.args.get("dataset_id")
    status = request.args.get("status", "active")

    sessions = []
    for s in _sessions.values():
        # Filter by dataset
        if dataset_id and s["dataset_id"] != dataset_id:
            continue

        # Filter by status
        if status and s["status"] != status:
            continue

        # Check if expired
        expires_at = datetime.fromisoformat(s["expires_at"])
        if expires_at < datetime.utcnow() and s["status"] == SessionStatus.ACTIVE.value:
            s["status"] = SessionStatus.ENDED.value

        sessions.append(
            {
                "id": s["id"],
                "name": s["name"],
                "dataset_id": s["dataset_id"],
                "status": s["status"],
                "participant_count": len(s["participants"]),
                "max_participants": s["max_participants"],
                "created_at": s["created_at"],
                "expires_at": s["expires_at"],
            }
        )

    return jsonify(
        {
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
        }
    )


@collaboration_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Get session details."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    session = _sessions[session_id]

    # Check if expired
    expires_at = datetime.fromisoformat(session["expires_at"])
    if (
        expires_at < datetime.utcnow()
        and session["status"] == SessionStatus.ACTIVE.value
    ):
        session["status"] = SessionStatus.ENDED.value

    # Get additional info
    comments = _session_comments.get(session_id, [])
    snapshots = _session_snapshots.get(session_id, [])

    return jsonify(
        {
            "success": True,
            "session": session,
            "comment_count": len(comments),
            "snapshot_count": len(snapshots),
            "active_cursors": len(_active_cursors.get(session_id, {})),
        }
    )


@collaboration_bp.route("/sessions/<session_id>", methods=["DELETE"])
def end_session(session_id: str):
    """End a collaboration session."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    _sessions[session_id]["status"] = SessionStatus.ENDED.value

    # Clear active cursors
    if session_id in _active_cursors:
        _active_cursors[session_id] = {}

    logger.info(f"Session ended: {session_id}")

    return jsonify(
        {
            "success": True,
            "message": "Session ended",
        }
    )


# =============================================================================
# Session Participation Endpoints
# =============================================================================


@collaboration_bp.route("/sessions/<session_id>/join", methods=["POST"])
def join_session(session_id: str):
    """
    Join an existing collaboration session.

    Request body:
        {
            "user_id": "uuid"
        }
    """
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    session = _sessions[session_id]

    # Check if session is active
    if session["status"] != SessionStatus.ACTIVE.value:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "SESSION_INACTIVE",
                    "message": "Session is not active",
                }
            ),
            400,
        )

    # Check if expired
    expires_at = datetime.fromisoformat(session["expires_at"])
    if expires_at < datetime.utcnow():
        session["status"] = SessionStatus.ENDED.value
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "SESSION_EXPIRED",
                    "message": "Session has expired",
                }
            ),
            400,
        )

    data = request.get_json() or {}
    user_id = data.get("user_id", str(uuid.uuid4()))

    # Check if already a participant
    if user_id in session["participants"]:
        return jsonify(
            {
                "success": True,
                "message": "Already a participant",
                "session": session,
            }
        )

    # Check capacity
    if len(session["participants"]) >= session["max_participants"]:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "SESSION_FULL",
                    "message": "Session is at capacity",
                }
            ),
            400,
        )

    # Add participant
    session["participants"].append(user_id)

    logger.info(f"User {user_id} joined session {session_id}")

    return jsonify(
        {
            "success": True,
            "message": "Joined session",
            "session": session,
        }
    )


@collaboration_bp.route("/sessions/<session_id>/leave", methods=["POST"])
def leave_session(session_id: str):
    """
    Leave a collaboration session.

    Request body:
        {
            "user_id": "uuid"
        }
    """
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    session = _sessions[session_id]

    data = request.get_json() or {}
    user_id = data.get("user_id")

    if not user_id:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "MISSING_FIELD",
                    "message": "user_id is required",
                }
            ),
            400,
        )

    if user_id in session["participants"]:
        session["participants"].remove(user_id)

    # Remove cursor
    if session_id in _active_cursors and user_id in _active_cursors[session_id]:
        del _active_cursors[session_id][user_id]

    logger.info(f"User {user_id} left session {session_id}")

    return jsonify(
        {
            "success": True,
            "message": "Left session",
        }
    )


@collaboration_bp.route("/sessions/<session_id>/participants", methods=["GET"])
def get_participants(session_id: str):
    """Get list of session participants."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    session = _sessions[session_id]

    # In a real app, you'd look up user details
    participants = [
        {
            "user_id": uid,
            "is_owner": uid == session["created_by"],
            "has_cursor": uid in _active_cursors.get(session_id, {}),
        }
        for uid in session["participants"]
    ]

    return jsonify(
        {
            "success": True,
            "participants": participants,
            "count": len(participants),
        }
    )


# =============================================================================
# Cursor Tracking Endpoints
# =============================================================================


@collaboration_bp.route("/sessions/<session_id>/cursors", methods=["GET"])
def get_cursors(session_id: str):
    """Get all active cursor positions in a session."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    cursors = _active_cursors.get(session_id, {})

    return jsonify(
        {
            "success": True,
            "cursors": list(cursors.values()),
            "count": len(cursors),
        }
    )


@collaboration_bp.route("/sessions/<session_id>/cursors", methods=["POST"])
def update_cursor(session_id: str):
    """
    Update cursor position.

    Request body:
        {
            "user_id": "uuid",
            "x": 0.5,
            "y": 0.3,
            "element_id": "optional"
        }
    """
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    data = request.get_json() or {}
    user_id = data.get("user_id")

    if not user_id:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "MISSING_FIELD",
                    "message": "user_id is required",
                }
            ),
            400,
        )

    cursor = {
        "user_id": user_id,
        "x": float(data.get("x", 0)),
        "y": float(data.get("y", 0)),
        "element_id": data.get("element_id"),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if session_id not in _active_cursors:
        _active_cursors[session_id] = {}

    _active_cursors[session_id][user_id] = cursor

    return jsonify(
        {
            "success": True,
            "cursor": cursor,
        }
    )


# =============================================================================
# Comment Endpoints
# =============================================================================


@collaboration_bp.route("/sessions/<session_id>/comments", methods=["GET"])
def get_comments(session_id: str):
    """Get all comments in a session."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    comments = _session_comments.get(session_id, [])

    # Optional filters
    insight_id = request.args.get("insight_id")
    visualization_id = request.args.get("visualization_id")

    if insight_id:
        comments = [c for c in comments if c.get("insight_id") == insight_id]

    if visualization_id:
        comments = [
            c for c in comments if c.get("visualization_id") == visualization_id
        ]

    return jsonify(
        {
            "success": True,
            "comments": comments,
            "count": len(comments),
        }
    )


@collaboration_bp.route("/sessions/<session_id>/comments", methods=["POST"])
def add_comment(session_id: str):
    """
    Add a comment to the session.

    Request body:
        {
            "user_id": "uuid",
            "content": "Comment text",
            "insight_id": "optional uuid",
            "visualization_id": "optional uuid",
            "parent_id": "optional uuid for replies"
        }
    """
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    data = request.get_json() or {}

    user_id = data.get("user_id")
    content = data.get("content", "").strip()

    if not user_id:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "MISSING_FIELD",
                    "message": "user_id is required",
                }
            ),
            400,
        )

    if not content:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "MISSING_FIELD",
                    "message": "content is required",
                }
            ),
            400,
        )

    if len(content) > 2000:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "CONTENT_TOO_LONG",
                    "message": "Comment must be 2000 characters or less",
                }
            ),
            400,
        )

    comment = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
        "content": content,
        "insight_id": data.get("insight_id"),
        "visualization_id": data.get("visualization_id"),
        "parent_id": data.get("parent_id"),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    _session_comments[session_id].append(comment)

    logger.info(f"Comment added to session {session_id}")

    return (
        jsonify(
            {
                "success": True,
                "comment": comment,
            }
        ),
        201,
    )


@collaboration_bp.route(
    "/sessions/<session_id>/comments/<comment_id>", methods=["DELETE"]
)
def delete_comment(session_id: str, comment_id: str):
    """Delete a comment."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    comments = _session_comments.get(session_id, [])

    # Find and remove comment
    for i, comment in enumerate(comments):
        if comment["id"] == comment_id:
            del comments[i]
            logger.info(f"Comment {comment_id} deleted from session {session_id}")
            return jsonify(
                {
                    "success": True,
                    "message": "Comment deleted",
                }
            )

    return (
        jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Comment not found",
            }
        ),
        404,
    )


# =============================================================================
# Snapshot Endpoints
# =============================================================================


@collaboration_bp.route("/sessions/<session_id>/snapshots", methods=["GET"])
def get_snapshots(session_id: str):
    """Get all snapshots in a session."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    snapshots = _session_snapshots.get(session_id, [])

    return jsonify(
        {
            "success": True,
            "snapshots": snapshots,
            "count": len(snapshots),
        }
    )


@collaboration_bp.route("/sessions/<session_id>/snapshots", methods=["POST"])
def create_snapshot(session_id: str):
    """
    Create a snapshot of the current session state.

    Request body:
        {
            "user_id": "uuid",
            "name": "Snapshot name",
            "description": "Optional description",
            "visualization_ids": ["list of viz ids to include"],
            "insight_ids": ["list of insight ids to include"]
        }
    """
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    data = request.get_json() or {}

    user_id = data.get("user_id")
    name = data.get(
        "name", f"Snapshot {len(_session_snapshots.get(session_id, [])) + 1}"
    )

    if not user_id:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "MISSING_FIELD",
                    "message": "user_id is required",
                }
            ),
            400,
        )

    snapshot = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "created_by": user_id,
        "name": name,
        "description": data.get("description", ""),
        "visualization_ids": data.get("visualization_ids", []),
        "insight_ids": data.get("insight_ids", []),
        "created_at": datetime.utcnow().isoformat(),
    }

    _session_snapshots[session_id].append(snapshot)

    logger.info(f"Snapshot created in session {session_id}")

    return (
        jsonify(
            {
                "success": True,
                "snapshot": snapshot,
            }
        ),
        201,
    )


@collaboration_bp.route(
    "/sessions/<session_id>/snapshots/<snapshot_id>", methods=["GET"]
)
def get_snapshot(session_id: str, snapshot_id: str):
    """Get a specific snapshot."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    snapshots = _session_snapshots.get(session_id, [])
    snapshot = next((s for s in snapshots if s["id"] == snapshot_id), None)

    if not snapshot:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Snapshot not found",
                }
            ),
            404,
        )

    return jsonify(
        {
            "success": True,
            "snapshot": snapshot,
        }
    )


@collaboration_bp.route(
    "/sessions/<session_id>/snapshots/<snapshot_id>", methods=["DELETE"]
)
def delete_snapshot(session_id: str, snapshot_id: str):
    """Delete a snapshot."""
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    snapshots = _session_snapshots.get(session_id, [])

    for i, snapshot in enumerate(snapshots):
        if snapshot["id"] == snapshot_id:
            del snapshots[i]
            logger.info(f"Snapshot {snapshot_id} deleted from session {session_id}")
            return jsonify(
                {
                    "success": True,
                    "message": "Snapshot deleted",
                }
            )

    return (
        jsonify(
            {
                "error": True,
                "error_code": "NOT_FOUND",
                "message": "Snapshot not found",
            }
        ),
        404,
    )


# =============================================================================
# Sharing Endpoints
# =============================================================================


@collaboration_bp.route("/sessions/<session_id>/share", methods=["POST"])
def share_to_session(session_id: str):
    """
    Share an insight or visualization to the session.

    Request body:
        {
            "user_id": "uuid",
            "type": "insight|visualization",
            "item_id": "uuid",
            "message": "Optional message"
        }
    """
    if session_id not in _sessions:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "NOT_FOUND",
                    "message": "Session not found",
                }
            ),
            404,
        )

    data = request.get_json() or {}

    user_id = data.get("user_id")
    share_type = data.get("type")
    item_id = data.get("item_id")

    if not all([user_id, share_type, item_id]):
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "MISSING_FIELDS",
                    "message": "user_id, type, and item_id are required",
                }
            ),
            400,
        )

    if share_type not in ["insight", "visualization"]:
        return (
            jsonify(
                {
                    "error": True,
                    "error_code": "INVALID_TYPE",
                    "message": "type must be 'insight' or 'visualization'",
                }
            ),
            400,
        )

    # Create a share event (would be broadcast via WebSocket in real app)
    share_event = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
        "type": share_type,
        "item_id": item_id,
        "message": data.get("message", ""),
        "created_at": datetime.utcnow().isoformat(),
    }

    logger.info(f"{share_type} shared in session {session_id}")

    return jsonify(
        {
            "success": True,
            "share_event": share_event,
        }
    )
