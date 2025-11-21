"""Pydantic schemas module."""
from .auth import (
    UserCreate,
    UserResponse,
    UserUpdate,
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
)
from .document import (
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
    DocumentChunkResponse,
    IngestRequest,
    IngestResponse,
)
from .query import (
    QueryRequest,
    QueryResponse,
    QuerySessionResponse,
)
from .agent import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentStepResponse,
)
from .common import (
    PaginatedResponse,
    ErrorResponse,
    HealthResponse,
)

__all__ = [
    # Auth
    "UserCreate",
    "UserResponse", 
    "UserUpdate",
    "LoginRequest",
    "TokenResponse",
    "RefreshTokenRequest",
    # Document
    "DocumentCreate",
    "DocumentResponse",
    "DocumentUpdate", 
    "DocumentChunkResponse",
    "IngestRequest",
    "IngestResponse",
    # Query
    "QueryRequest",
    "QueryResponse",
    "QuerySessionResponse",
    # Agent
    "AgentExecutionRequest",
    "AgentExecutionResponse",
    "AgentStepResponse",
    # Common
    "PaginatedResponse",
    "ErrorResponse",
    "HealthResponse",
]