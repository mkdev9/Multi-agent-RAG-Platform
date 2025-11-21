"""Document management routes."""
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...schemas.document import (
    DocumentResponse, IngestRequest, IngestResponse, 
    DocumentSearchRequest, ProcessingJob
)
from ...schemas.common import PaginatedResponse, SuccessResponse, PaginationParams
from ...core.security import get_current_user
from ...core.exceptions import NotFoundError, ProcessingError
from ...services.document import document_service
from ...models.user import User
import tempfile
import os

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form("{}"),
    tags: Optional[str] = Form("[]"),
    processing_options: Optional[str] = Form("{}"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Ingest a document file."""
    try:
        import json
        
        # Parse form data
        metadata_dict = json.loads(metadata) if metadata else {}
        tags_list = json.loads(tags) if tags else []
        options_dict = json.loads(processing_options) if processing_options else {}
        
        # Create ingest request
        ingest_request = IngestRequest(
            metadata=metadata_dict,
            tags=tags_list,
            processing_options=options_dict
        )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Ingest file
            document = await document_service.ingest_file(
                db=db,
                file_path=temp_file_path,
                filename=file.filename,
                user=current_user,
                ingest_request=ingest_request
            )
            
            return IngestResponse(
                job_id=document.id,
                document_id=document.id,
                status=document.processing_status,
                message="Document ingestion started"
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise ProcessingError(f"Failed to ingest document: {str(e)}")


@router.post("/ingest-url", response_model=IngestResponse)
async def ingest_url(
    url: str,
    ingest_request: IngestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Ingest content from URL."""
    try:
        document = await document_service.ingest_url(
            db=db,
            url=url,
            user=current_user,
            ingest_request=ingest_request
        )
        
        return IngestResponse(
            job_id=document.id,
            document_id=document.id,
            status=document.processing_status,
            message="URL ingestion started"
        )
    
    except Exception as e:
        raise ProcessingError(f"Failed to ingest URL: {str(e)}")


@router.get("", response_model=PaginatedResponse[DocumentResponse])
async def get_documents(
    pagination: PaginationParams = Depends(),
    file_types: Optional[List[str]] = Query(None),
    status: Optional[List[str]] = Query(None),
    tags: Optional[List[str]] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's documents with pagination and filtering."""
    # Build filters
    filters = {}
    if file_types:
        filters["file_types"] = file_types
    if status:
        filters["status"] = status
    if tags:
        filters["tags"] = tags
    
    # Get documents
    documents, total = await document_service.get_documents(
        db=db,
        user=current_user,
        skip=pagination.offset,
        limit=pagination.size,
        filters=filters
    )
    
    # Convert to response models
    document_responses = [
        DocumentResponse.model_validate(doc) for doc in documents
    ]
    
    return PaginatedResponse.create(
        items=document_responses,
        total=total,
        page=pagination.page,
        size=pagination.size
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get document by ID."""
    document = await document_service.get_document_by_id(
        db=db,
        document_id=document_id,
        user=current_user
    )
    
    if not document:
        raise NotFoundError("Document not found")
    
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", response_model=SuccessResponse)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete document and all associated chunks."""
    success = await document_service.delete_document(
        db=db,
        document_id=document_id,
        user=current_user
    )
    
    if not success:
        raise NotFoundError("Document not found")
    
    return SuccessResponse(message="Document deleted successfully")


@router.get("/{document_id}/status", response_model=ProcessingJob)
async def get_processing_status(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get document processing status."""
    document = await document_service.get_document_by_id(
        db=db,
        document_id=document_id,
        user=current_user
    )
    
    if not document:
        raise NotFoundError("Document not found")
    
    # Calculate progress based on status
    progress_map = {
        "pending": 0.0,
        "processing": 50.0,
        "completed": 100.0,
        "failed": 0.0
    }
    
    return ProcessingJob(
        job_id=document.id,
        document_id=document.id,
        status=document.processing_status,
        progress=progress_map.get(document.processing_status, 0.0),
        error_message=document.processing_error,
        started_at=document.created_at,
        completed_at=document.updated_at if document.processing_status in ["completed", "failed"] else None
    )