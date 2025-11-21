"""Query models."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import String, Integer, Float, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class QuerySession(Base):
    """Query session model for tracking user sessions."""
    
    __tablename__ = "query_sessions"
    
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )
    
    session_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    session_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    
    total_queries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    
    # Session metadata
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    session_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Relationships
    queries: Mapped[List["Query"]] = relationship(
        "Query",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<QuerySession(user_id={self.user_id}, queries={self.total_queries})>"


class Query(Base):
    """Query model for tracking individual questions and responses."""
    
    __tablename__ = "queries"
    
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_sessions.id"),
        nullable=False,
        index=True
    )
    
    # Query content
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text)
    
    # Query processing info
    query_type: Mapped[str] = mapped_column(
        String(50), 
        default="rag",
        nullable=False
    )  # rag, agent_workflow, direct
    
    # Performance metrics
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    token_count_input: Mapped[Optional[int]] = mapped_column(Integer)
    token_count_output: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Quality metrics
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float)
    user_feedback_rating: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5
    
    # Sources and context
    sources_used: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)  # chunk IDs, scores
    context_chunks: Mapped[Optional[List[str]]] = mapped_column(JSONB)  # chunk content
    
    # Processing details
    llm_provider: Mapped[Optional[str]] = mapped_column(String(50))
    llm_model: Mapped[Optional[str]] = mapped_column(String(100))
    retrieval_method: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Error handling
    status: Mapped[str] = mapped_column(
        String(20),
        default="completed",
        nullable=False
    )  # processing, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Additional metadata
    filters_applied: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    query_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Relationships
    session: Mapped[QuerySession] = relationship(
        "QuerySession",
        back_populates="queries"
    )
    
    def __repr__(self) -> str:
        return f"<Query(id={self.id}, type={self.query_type}, status={self.status})>"