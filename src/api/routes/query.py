"""Query and RAG routes."""
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, Query as QueryParam
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...schemas.query import (
    QueryRequest, QueryResponse, QuerySessionResponse,
    QueryFeedback, QueryStats, SimilarQuestionsResponse
)
from ...schemas.common import PaginatedResponse, SuccessResponse, PaginationParams
from ...core.security import get_current_user
from ...core.exceptions import NotFoundError
from ...services.query import query_service
from ...models.user import User

router = APIRouter(prefix="/query", tags=["Query & RAG"])


@router.post("", response_model=QueryResponse)
async def process_query(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Process a natural language query using RAG."""
    response = await query_service.process_query(
        db=db,
        user=current_user,
        query_request=query_request
    )
    
    return response


@router.get("/history", response_model=PaginatedResponse[QueryResponse])
async def get_query_history(
    pagination: PaginationParams = Depends(),
    session_id: Optional[uuid.UUID] = QueryParam(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's query history."""
    queries, total = await query_service.get_query_history(
        db=db,
        user=current_user,
        session_id=session_id,
        skip=pagination.offset,
        limit=pagination.size
    )
    
    # Convert to response models
    query_responses = []
    for query in queries:
        # Build sources from stored data
        sources = []
        if query.sources_used and "chunks" in query.sources_used:
            # In a full implementation, you'd fetch the actual chunk data
            # For now, we'll return minimal source info
            for chunk_info in query.sources_used["chunks"][:5]:
                sources.append({
                    "chunk_id": chunk_info["chunk_id"],
                    "score": chunk_info["score"],
                    "content": "Content not available in history",
                    "document_name": "Unknown"
                })
        
        query_response = QueryResponse(
            query_id=query.id,
            question=query.question,
            answer=query.answer or "",
            confidence_score=query.confidence_score or 0.0,
            sources=[],  # Simplified for history
            response_time_ms=query.response_time_ms or 0,
            token_count_input=query.token_count_input or 0,
            token_count_output=query.token_count_output or 0,
            llm_provider=query.llm_provider or "unknown",
            llm_model=query.llm_model or "unknown",
            filters_applied=query.filters_applied,
            query_metadata=query.query_metadata,
            created_at=query.created_at
        )
        query_responses.append(query_response)
    
    return PaginatedResponse.create(
        items=query_responses,
        total=total,
        page=pagination.page,
        size=pagination.size
    )


@router.get("/{query_id}", response_model=QueryResponse)
async def get_query(
    query_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific query by ID."""
    query = await query_service.get_query_by_id(
        db=db,
        query_id=query_id,
        user=current_user
    )
    
    if not query:
        raise NotFoundError("Query not found")
    
    # Build response (simplified)
    return QueryResponse(
        query_id=query.id,
        question=query.question,
        answer=query.answer or "",
        confidence_score=query.confidence_score or 0.0,
        sources=[],  # Would need to reconstruct from stored data
        response_time_ms=query.response_time_ms or 0,
        token_count_input=query.token_count_input or 0,
        token_count_output=query.token_count_output or 0,
        llm_provider=query.llm_provider or "unknown",
        llm_model=query.llm_model or "unknown",
        filters_applied=query.filters_applied,
        query_metadata=query.query_metadata,
        created_at=query.created_at
    )


@router.post("/{query_id}/feedback", response_model=SuccessResponse)
async def submit_query_feedback(
    query_id: uuid.UUID,
    feedback: QueryFeedback,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback for a query."""
    success = await query_service.update_query_feedback(
        db=db,
        query_id=query_id,
        user=current_user,
        rating=feedback.rating,
        feedback_text=feedback.feedback_text
    )
    
    if not success:
        raise NotFoundError("Query not found")
    
    return SuccessResponse(message="Feedback submitted successfully")


@router.get("/sessions/list", response_model=PaginatedResponse[QuerySessionResponse])
async def get_query_sessions(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's query sessions."""
    sessions, total = await query_service.get_user_sessions(
        db=db,
        user=current_user,
        skip=pagination.offset,
        limit=pagination.size
    )
    
    # Convert to response models
    session_responses = [
        QuerySessionResponse(
            id=session.id,
            user_id=session.user_id,
            session_start=session.session_start,
            session_end=session.session_end,
            total_queries=session.total_queries,
            total_tokens_used=session.total_tokens_used
        )
        for session in sessions
    ]
    
    return PaginatedResponse.create(
        items=session_responses,
        total=total,
        page=pagination.page,
        size=pagination.size
    )


@router.post("/sessions/{session_id}/end", response_model=SuccessResponse)
async def end_query_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """End a query session."""
    success = await query_service.end_session(
        db=db,
        session_id=session_id,
        user=current_user
    )
    
    if not success:
        raise NotFoundError("Session not found or already ended")
    
    return SuccessResponse(message="Session ended successfully")


@router.get("/suggestions/generate", response_model=List[str])
async def get_query_suggestions(
    context: Optional[str] = QueryParam(None),
    limit: int = QueryParam(5, ge=1, le=10),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get query suggestions for the user."""
    suggestions = await query_service.get_query_suggestions(
        db=db,
        user=current_user,
        context=context,
        limit=limit
    )
    
    return suggestions


@router.post("/similar", response_model=SimilarQuestionsResponse)
async def find_similar_questions(
    question: str,
    limit: int = QueryParam(5, ge=1, le=10),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Find similar questions from user's query history."""
    similar_questions = await query_service.get_similar_questions(
        db=db,
        question=question,
        user=current_user,
        limit=limit
    )
    
    return SimilarQuestionsResponse(
        original_question=question,
        similar_questions=similar_questions
    )