"""Document models."""
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import String, Integer, BigInteger, Text, ForeignKey, ARRAY, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Document(Base):
    """Document model for uploaded files."""
    
    __tablename__ = "documents"
    
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    file_path: Mapped[Optional[str]] = mapped_column(Text)  # Storage path
    
    # Processing status
    processing_status: Mapped[str] = mapped_column(
        String(50), 
        default="pending", 
        nullable=False
    )  # pending, processing, completed, failed
    processing_error: Mapped[Optional[str]] = mapped_column(Text)
    
    # Metadata and organization
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    
    # User who uploaded
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("users.id"),
        nullable=False
    )
    
    # Content statistics
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Document(filename={self.filename}, status={self.processing_status})>"


class DocumentChunk(Base):
    """Document chunk model for processed text segments."""
    
    __tablename__ = "document_chunks"
    
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Chunk information
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    
    # Token and size information
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Vector database reference
    embedding_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Position in original document
    start_char: Mapped[Optional[int]] = mapped_column(Integer)
    end_char: Mapped[Optional[int]] = mapped_column(Integer)
    page_number: Mapped[Optional[int]] = mapped_column(Integer)
    section_title: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Chunk metadata
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Relationships
    document: Mapped[Document] = relationship(
        "Document",
        back_populates="chunks"
    )
    
    def __repr__(self) -> str:
        return f"<DocumentChunk(doc_id={self.document_id}, index={self.chunk_index})>"