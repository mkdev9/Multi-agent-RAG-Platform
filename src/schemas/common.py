"""Common Pydantic schemas."""
from typing import Any, Dict, Generic, List, Optional, TypeVar
from datetime import datetime

from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = {
        "from_attributes": True,
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


class PaginatedResponse(BaseSchema, Generic[T]):
    """Generic paginated response."""
    
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        size: int,
    ) -> "PaginatedResponse[T]":
        """Create paginated response."""
        pages = (total + size - 1) // size if size > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
        )


class ErrorResponse(BaseSchema):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseSchema):
    """Generic success response."""
    
    success: bool = Field(True, description="Success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class HealthResponse(BaseSchema):
    """Health check response."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Application version")
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Service health statuses"
    )
    metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Health metrics"
    )


class MetricsResponse(BaseSchema):
    """Metrics response schema."""
    
    total_queries: int = Field(..., description="Total number of queries")
    total_documents: int = Field(..., description="Total number of documents")
    total_users: int = Field(..., description="Total number of users")
    avg_response_time_ms: float = Field(..., description="Average response time")
    queries_per_day: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Daily query statistics"
    )
    top_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most used document sources"
    )
    
    
class FilterParams(BaseSchema):
    """Common filter parameters."""
    
    search: Optional[str] = Field(None, description="Search term")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    source: Optional[str] = Field(None, description="Filter by source")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")


class PaginationParams(BaseSchema):
    """Pagination parameters."""
    
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Page size")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size