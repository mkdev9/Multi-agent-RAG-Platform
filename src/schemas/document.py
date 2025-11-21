"""Document schemas."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from .common import BaseSchema


class DocumentBase(BaseSchema):
    """Base document schema."""
    
    filename: str = Field(..., description="Document filename")
    file_type: str = Field(..., description="Document file type")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")


class DocumentCreate(DocumentBase):
    """Document creation schema."""
    
    source_url: Optional[str] = Field(None, description="Source URL if from web")


class DocumentUpdate(BaseSchema):
    """Document update schema."""
    
    tags: Optional[List[str]] = Field(None, description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")


class DocumentResponse(DocumentBase):
    """Document response schema."""
    
    id: uuid.UUID = Field(..., description="Document ID")
    original_filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type")
    source_url: Optional[str] = Field(None, description="Source URL")
    processing_status: str = Field(..., description="Processing status")
    processing_error: Optional[str] = Field(None, description="Processing error message")
    total_chunks: int = Field(..., description="Number of chunks created")
    total_tokens: int = Field(..., description="Total token count")
    uploaded_by: uuid.UUID = Field(..., description="User who uploaded")
    created_at: datetime = Field(..., description="Upload timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class DocumentChunkResponse(BaseSchema):
    """Document chunk response schema."""
    
    id: uuid.UUID = Field(..., description="Chunk ID")
    document_id: uuid.UUID = Field(..., description="Document ID")
    chunk_index: int = Field(..., description="Chunk index")
    content: str = Field(..., description="Chunk content")
    token_count: int = Field(..., description="Token count")
    char_count: int = Field(..., description="Character count")
    embedding_id: Optional[str] = Field(None, description="Vector DB embedding ID")
    start_char: Optional[int] = Field(None, description="Start character position")
    end_char: Optional[int] = Field(None, description="End character position")
    page_number: Optional[int] = Field(None, description="Page number")
    section_title: Optional[str] = Field(None, description="Section title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Chunk metadata")
    created_at: datetime = Field(..., description="Creation timestamp")


class IngestRequest(BaseSchema):
    """Document ingestion request schema."""
    
    processing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Processing options"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Document tags"
    )
    
    @validator('processing_options')
    def validate_processing_options(cls, v):
        """Validate processing options."""
        if v is None:
            return {}
        
        allowed_keys = {
            'chunk_size', 'chunk_overlap', 'embedding_model',
            'extract_tables', 'extract_images', 'ocr_enabled'
        }
        
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid processing option: {key}")
        
        return v


class IngestResponse(BaseSchema):
    """Document ingestion response schema."""
    
    job_id: uuid.UUID = Field(..., description="Processing job ID")
    document_id: uuid.UUID = Field(..., description="Document ID")
    status: str = Field(..., description="Initial processing status")
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )
    message: str = Field(..., description="Status message")


class DocumentSearchRequest(BaseSchema):
    """Document search request schema."""
    
    query: Optional[str] = Field(None, description="Search query")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    file_types: Optional[List[str]] = Field(None, description="Filter by file types")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    status: Optional[List[str]] = Field(None, description="Filter by processing status")
    uploaded_by: Optional[uuid.UUID] = Field(None, description="Filter by uploader")


class ProcessingJob(BaseSchema):
    """Processing job status schema."""
    
    job_id: uuid.UUID = Field(..., description="Job ID")
    document_id: uuid.UUID = Field(..., description="Document ID")
    status: str = Field(..., description="Processing status")
    progress: float = Field(..., description="Processing progress (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: datetime = Field(..., description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    
    
class BulkIngestRequest(BaseSchema):
    """Bulk document ingestion request."""
    
    source_urls: List[str] = Field(..., description="List of URLs to ingest")
    processing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Processing options"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Tags to apply to all documents"
    )
    
    
class BulkIngestResponse(BaseSchema):
    """Bulk document ingestion response."""
    
    batch_id: uuid.UUID = Field(..., description="Batch processing ID")
    jobs: List[IngestResponse] = Field(..., description="Individual job responses")
    total_documents: int = Field(..., description="Total number of documents")
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated batch completion time"
    )