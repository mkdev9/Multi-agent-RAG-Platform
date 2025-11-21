"""Query schemas."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from .common import BaseSchema


class QueryRequest(BaseSchema):
    """Query request schema."""
    
    question: str = Field(..., description="Natural language question")
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Search filters"
    )
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Query options"
    )
    
    @validator('options')
    def validate_options(cls, v):
        """Validate query options."""
        if v is None:
            return {}
        
        # Set defaults
        defaults = {
            'max_chunks': 10,
            'include_metadata': True,
            'similarity_threshold': 0.7,
            'rerank': True,
            'temperature': 0.7,
            'max_tokens': 1000,
        }
        
        # Merge with defaults
        for key, default_value in defaults.items():
            if key not in v:
                v[key] = default_value
        
        return v


class SourceReference(BaseSchema):
    """Source reference schema."""
    
    chunk_id: uuid.UUID = Field(..., description="Document chunk ID")
    document_id: uuid.UUID = Field(..., description="Document ID")
    document_name: str = Field(..., description="Document name")
    content: str = Field(..., description="Relevant content")
    score: float = Field(..., description="Relevance score")
    page_number: Optional[int] = Field(None, description="Page number")
    section_title: Optional[str] = Field(None, description="Section title")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Source metadata"
    )


class QueryResponse(BaseSchema):
    """Query response schema."""
    
    query_id: uuid.UUID = Field(..., description="Query ID")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Answer confidence score")
    relevance_score: Optional[float] = Field(None, description="Overall relevance score")
    
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Source references"
    )
    
    # Metadata
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    token_count_input: int = Field(..., description="Input token count")
    token_count_output: int = Field(..., description="Output token count")
    llm_provider: str = Field(..., description="LLM provider used")
    llm_model: str = Field(..., description="LLM model used")
    
    # Additional info
    filters_applied: Optional[Dict[str, Any]] = Field(
        None,
        description="Filters that were applied"
    )
    query_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional query metadata"
    )
    
    created_at: datetime = Field(..., description="Query timestamp")


class QuerySessionResponse(BaseSchema):
    """Query session response schema."""
    
    id: uuid.UUID = Field(..., description="Session ID")
    user_id: uuid.UUID = Field(..., description="User ID")
    session_start: datetime = Field(..., description="Session start time")
    session_end: Optional[datetime] = Field(None, description="Session end time")
    total_queries: int = Field(..., description="Total queries in session")
    total_tokens_used: int = Field(..., description="Total tokens used")
    queries: Optional[List[QueryResponse]] = Field(
        None,
        description="Queries in session"
    )


class QueryFeedback(BaseSchema):
    """Query feedback schema."""
    
    query_id: uuid.UUID = Field(..., description="Query ID")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    feedback_text: Optional[str] = Field(None, description="Feedback text")
    helpful_sources: Optional[List[uuid.UUID]] = Field(
        None,
        description="Helpful source chunk IDs"
    )
    suggestions: Optional[str] = Field(None, description="Improvement suggestions")


class QueryStats(BaseSchema):
    """Query statistics schema."""
    
    total_queries: int = Field(..., description="Total number of queries")
    avg_response_time_ms: float = Field(..., description="Average response time")
    avg_confidence_score: float = Field(..., description="Average confidence score")
    most_common_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most common query patterns"
    )
    queries_by_hour: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Query distribution by hour"
    )
    top_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most referenced sources"
    )


class SimilarQuestionsResponse(BaseSchema):
    """Similar questions response schema."""
    
    original_question: str = Field(..., description="Original question")
    similar_questions: List[Dict[str, Any]] = Field(
        ...,
        description="Similar questions with scores"
    )
    
    
class QuerySuggestion(BaseSchema):
    """Query suggestion schema."""
    
    suggestion: str = Field(..., description="Suggested question")
    score: float = Field(..., description="Relevance score")
    category: Optional[str] = Field(None, description="Question category")
    

class QuerySuggestionsResponse(BaseSchema):
    """Query suggestions response schema."""
    
    suggestions: List[QuerySuggestion] = Field(
        ...,
        description="Query suggestions"
    )
    based_on: str = Field(..., description="What suggestions are based on")