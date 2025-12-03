"""
User and collaboration data models.

This module defines Pydantic models for users, sessions, and collaboration.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """User roles in the system."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


class User(BaseModel):
    """
    A user in the InsightBoost system.

    Attributes:
        id: Unique identifier
        email: User's email address
        name: Display name
        role: User's role
        created_at: Account creation time
        last_login: Last login time
        is_active: Whether account is active
        preferences: User preferences
        avatar_url: URL to user's avatar
    """

    id: UUID = Field(default_factory=uuid4)
    email: EmailStr
    name: str = Field(min_length=1, max_length=100)
    role: UserRole = UserRole.ANALYST
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime | None = None
    is_active: bool = True
    preferences: dict[str, Any] = Field(default_factory=dict)
    avatar_url: str | None = None

    @property
    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return self.role == UserRole.ADMIN

    @property
    def can_analyze(self) -> bool:
        """Check if user can perform analysis."""
        return self.role in (UserRole.ANALYST, UserRole.ADMIN)

    def to_public_dict(self) -> dict[str, Any]:
        """Convert to dictionary safe for public display."""
        return {
            "id": str(self.id),
            "name": self.name,
            "role": self.role.value,
            "avatar_url": self.avatar_url,
        }

    def to_profile_dict(self) -> dict[str, Any]:
        """Convert to dictionary for user profile."""
        return {
            "id": str(self.id),
            "email": self.email,
            "name": self.name,
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "avatar_url": self.avatar_url,
        }


class SessionStatus(str, Enum):
    """Collaboration session status."""

    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class CollaborationSession(BaseModel):
    """
    A real-time collaboration session.

    Attributes:
        id: Unique identifier
        name: Session name
        dataset_id: Dataset being analyzed
        created_by: User who created the session
        created_at: When session was created
        status: Current session status
        participants: User IDs of participants
        max_participants: Maximum allowed participants
        expires_at: When session automatically ends
        settings: Session settings
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=100)
    dataset_id: UUID
    created_by: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: SessionStatus = SessionStatus.ACTIVE
    participants: list[UUID] = Field(default_factory=list)
    max_participants: int = Field(default=10, ge=2, le=50)
    expires_at: datetime | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    @property
    def participant_count(self) -> int:
        """Get current participant count."""
        return len(self.participants)

    @property
    def is_full(self) -> bool:
        """Check if session is at capacity."""
        return self.participant_count >= self.max_participants

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        if self.status != SessionStatus.ACTIVE:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def can_join(self, user_id: UUID) -> bool:
        """Check if user can join this session."""
        if user_id in self.participants:
            return True  # Already a participant
        return self.is_active and not self.is_full

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "dataset_id": str(self.dataset_id),
            "status": self.status.value,
            "participant_count": self.participant_count,
            "max_participants": self.max_participants,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
        }


class SessionEvent(BaseModel):
    """
    An event in a collaboration session.

    Attributes:
        id: Unique identifier
        session_id: Session this event belongs to
        event_type: Type of event
        user_id: User who triggered the event
        timestamp: When event occurred
        data: Event-specific data
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    event_type: str
    user_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = Field(default_factory=dict)


class SessionComment(BaseModel):
    """
    A comment in a collaboration session.

    Attributes:
        id: Unique identifier
        session_id: Session this comment belongs to
        user_id: User who made the comment
        insight_id: Insight being commented on (optional)
        visualization_id: Visualization being commented on (optional)
        content: Comment text
        created_at: When comment was created
        updated_at: When comment was last updated
        parent_id: Parent comment ID for replies
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID
    insight_id: UUID | None = None
    visualization_id: UUID | None = None
    content: str = Field(min_length=1, max_length=2000)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    parent_id: UUID | None = None

    @property
    def is_reply(self) -> bool:
        """Check if this is a reply to another comment."""
        return self.parent_id is not None


class CursorPosition(BaseModel):
    """
    A user's cursor position for shared exploration.

    Attributes:
        user_id: User whose cursor this is
        session_id: Session this cursor is in
        x: X coordinate (normalized 0-1)
        y: Y coordinate (normalized 0-1)
        element_id: ID of element cursor is over (optional)
        timestamp: When position was updated
    """

    user_id: UUID
    session_id: UUID
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    element_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIKey(BaseModel):
    """
    An API key for programmatic access.

    Attributes:
        id: Unique identifier
        user_id: User who owns this key
        name: Key name/description
        key_prefix: First few characters of key (for identification)
        key_hash: Hashed key value
        created_at: When key was created
        last_used: When key was last used
        expires_at: When key expires
        scopes: Allowed scopes for this key
        is_active: Whether key is active
    """

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    name: str = Field(min_length=1, max_length=100)
    key_prefix: str = Field(min_length=4, max_length=10)
    key_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: datetime | None = None
    expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list)
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if key is valid for use."""
        return self.is_active and not self.is_expired
