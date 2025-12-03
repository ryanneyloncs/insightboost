"""
Dataset data models.

This module defines Pydantic models for datasets and their metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class ColumnDataType(str, Enum):
    """Column data types."""

    NUMERIC = "numeric"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


class ColumnInfo(BaseModel):
    """
    Information about a dataset column.

    Attributes:
        name: Column name
        data_type: Detected data type
        original_dtype: Original pandas dtype string
        null_count: Number of null values
        null_percentage: Percentage of null values
        unique_count: Number of unique values
        unique_percentage: Percentage unique
        sample_values: Sample values from the column
        statistics: Statistical summary (for numeric columns)
        is_potential_id: Whether column looks like an ID field
        is_potential_date: Whether column could be parsed as date
    """

    name: str = Field(min_length=1, max_length=256)
    data_type: ColumnDataType
    original_dtype: str
    null_count: int = Field(ge=0)
    null_percentage: float = Field(ge=0.0, le=100.0)
    unique_count: int = Field(ge=0)
    unique_percentage: float = Field(ge=0.0, le=100.0)
    sample_values: list[Any] = Field(default_factory=list, max_length=10)
    statistics: dict[str, Any] = Field(default_factory=dict)
    is_potential_id: bool = False
    is_potential_date: bool = False

    @field_validator("null_percentage", "unique_percentage")
    @classmethod
    def round_percentage(cls, v: float) -> float:
        """Round percentages to 2 decimal places."""
        return round(v, 2)

    @property
    def is_complete(self) -> bool:
        """Check if column has no null values."""
        return self.null_count == 0

    @property
    def cardinality(self) -> str:
        """Get cardinality classification."""
        if self.unique_percentage == 100:
            return "unique"
        elif self.unique_percentage > 50:
            return "high"
        elif self.unique_percentage > 10:
            return "medium"
        else:
            return "low"


class DatasetMetadata(BaseModel):
    """
    Metadata about a dataset.

    Attributes:
        file_name: Original file name
        file_size_bytes: File size in bytes
        file_format: File format (csv, xlsx, json, etc.)
        encoding: File encoding (if known)
        delimiter: CSV delimiter (if applicable)
        has_header: Whether file has header row
        upload_source: Where the file came from
    """

    file_name: str
    file_size_bytes: int = Field(ge=0)
    file_format: str
    encoding: str = "utf-8"
    delimiter: str | None = None
    has_header: bool = True
    upload_source: str = "upload"

    @property
    def file_size_mb(self) -> float:
        """Get file size in MB."""
        return round(self.file_size_bytes / (1024 * 1024), 2)


class Dataset(BaseModel):
    """
    A dataset stored in InsightBoost.

    Attributes:
        id: Unique identifier
        name: Display name for the dataset
        description: Optional description
        owner_id: ID of the user who owns this dataset
        columns: Column information
        row_count: Number of rows
        created_at: When dataset was uploaded
        updated_at: When dataset was last modified
        last_analyzed: When dataset was last analyzed
        storage_path: Path to stored data file
        metadata: File metadata
        tags: User-defined tags
        is_public: Whether dataset is publicly accessible
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    owner_id: UUID
    columns: list[ColumnInfo] = Field(default_factory=list)
    row_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_analyzed: datetime | None = None
    storage_path: str
    metadata: DatasetMetadata
    tags: list[str] = Field(default_factory=list, max_length=20)
    is_public: bool = False

    @property
    def column_count(self) -> int:
        """Get number of columns."""
        return len(self.columns)

    @property
    def column_names(self) -> list[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]

    @property
    def numeric_columns(self) -> list[ColumnInfo]:
        """Get numeric columns."""
        return [
            col
            for col in self.columns
            if col.data_type
            in (ColumnDataType.NUMERIC, ColumnDataType.INTEGER, ColumnDataType.FLOAT)
        ]

    @property
    def categorical_columns(self) -> list[ColumnInfo]:
        """Get categorical columns."""
        return [
            col for col in self.columns if col.data_type == ColumnDataType.CATEGORICAL
        ]

    @property
    def datetime_columns(self) -> list[ColumnInfo]:
        """Get datetime columns."""
        return [col for col in self.columns if col.data_type == ColumnDataType.DATETIME]

    def get_column(self, name: str) -> ColumnInfo | None:
        """Get column info by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "created_at": self.created_at.isoformat(),
            "last_analyzed": (
                self.last_analyzed.isoformat() if self.last_analyzed else None
            ),
            "file_format": self.metadata.file_format,
            "file_size_mb": self.metadata.file_size_mb,
            "tags": self.tags,
        }

    def to_detail_dict(self) -> dict[str, Any]:
        """Convert to detailed dictionary for API responses."""
        summary = self.to_summary_dict()
        summary["columns"] = [col.model_dump() for col in self.columns]
        summary["metadata"] = self.metadata.model_dump()
        return summary


class DatasetVersion(BaseModel):
    """
    A version of a dataset.

    Attributes:
        id: Unique identifier
        dataset_id: Parent dataset ID
        version_number: Version number (1, 2, 3, ...)
        storage_path: Path to this version's data
        row_count: Rows in this version
        created_at: When version was created
        created_by: User who created this version
        change_description: Description of changes
    """

    id: UUID = Field(default_factory=uuid4)
    dataset_id: UUID
    version_number: int = Field(ge=1)
    storage_path: str
    row_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: UUID
    change_description: str | None = Field(default=None, max_length=500)


class DatasetShareSettings(BaseModel):
    """
    Sharing settings for a dataset.

    Attributes:
        dataset_id: Dataset being shared
        is_public: Publicly accessible
        shared_with_users: User IDs with access
        shared_with_teams: Team IDs with access
        allow_download: Whether users can download
        allow_copy: Whether users can copy
        expires_at: When sharing expires
    """

    dataset_id: UUID
    is_public: bool = False
    shared_with_users: list[UUID] = Field(default_factory=list)
    shared_with_teams: list[UUID] = Field(default_factory=list)
    allow_download: bool = True
    allow_copy: bool = False
    expires_at: datetime | None = None

    @property
    def is_shared(self) -> bool:
        """Check if dataset is shared with anyone."""
        return (
            self.is_public
            or bool(self.shared_with_users)
            or bool(self.shared_with_teams)
        )
