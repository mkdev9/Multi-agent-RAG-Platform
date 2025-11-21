"""Query service for RAG-based question answering."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..config import settings
from ..models.user import User
from ..models.query import Query, QuerySession
from ..schemas.query import QueryRequest, QueryResponse, SourceReference
from ..core.exceptions import LLMError, VectorStoreError
from ..core.logging import BusinessLogger
from .llm import llm_service
from .vector_store import vector_store_service

logger = BusinessLogger()


class PromptTemplate:
    """Prompt templates for different query types."""
    
    RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context. 
Use only the information from the context to answer questions. If the context doesn't contain enough 
information to answer the question, say so clearly.

Guidelines:
- Be accurate and precise
- Use direct quotes when possible
- Cite sources when appropriate
- If uncertain, express your level of confidence
- Do not make up information not present in the context"""

    RAG_USER_PROMPT = """Context information:
{context}

Question: {question}

Please provide a detailed answer based on the context above. If you reference specific information, 
indicate which source(s) it comes from."""

    SUMMARIZATION_PROMPT = """Please summarize the following text, highlighting the key points:

{text}

Summary:"""

    @classmethod
    def build_rag_prompt(cls, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Build RAG prompt with context."""
        # Format context chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = f"Source {i}"
            if chunk.get("document_name"):
                source_info += f" ({chunk['document_name']}"
                if chunk.get("page_number"):
                    source_info += f", page {chunk['page_number']}"
                source_info += ")"
            
            context_parts.append(f"{source_info}:\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        
        return f"{cls.RAG_SYSTEM_PROMPT}\n\n{cls.RAG_USER_PROMPT.format(context=context, question=question)}"


class QueryService:
    """Query processing and RAG service."""
    
    def __init__(self):
        self.prompt_template = PromptTemplate()
    
    async def create_or_get_session(
        self,
        db: AsyncSession,
        user: User,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> QuerySession:
        """Create or get current query session."""
        # Get or create active session for user
        stmt = (
            select(QuerySession)
            .where(QuerySession.user_id == user.id)
            .where(QuerySession.session_end.is_(None))
            .order_by(QuerySession.session_start.desc())
        )
        
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session:
            session = QuerySession(
                user_id=user.id,
                session_start=datetime.utcnow(),
                session_metadata=session_metadata or {}
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
        
        return session
    
    async def process_query(
        self,
        db: AsyncSession,
        user: User,
        query_request: QueryRequest,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """Process a RAG query."""
        start_time = datetime.utcnow()
        
        try:
            # Get or create session
            session = await self.create_or_get_session(db, user, session_metadata)
            
            # Create query record
            query = Query(
                session_id=session.id,
                question=query_request.question,
                query_type="rag",
                filters_applied=query_request.filters,
                query_metadata={"options": query_request.options}
            )
            db.add(query)
            await db.commit()
            await db.refresh(query)
            
            # Generate query embedding
            query_embedding = await llm_service.generate_embedding(
                text=query_request.question
            )
            
            # Search for relevant chunks
            search_results = await vector_store_service.search_similar_chunks(
                query_embedding=query_embedding,
                k=query_request.options.get("max_chunks", 10),
                filters=query_request.filters,
                score_threshold=query_request.options.get("similarity_threshold", 0.7)
            )
            
            if not search_results:
                # No relevant context found
                answer = "I don't have enough relevant information in my knowledge base to answer this question."
                confidence_score = 0.0
                sources = []
            else:
                # Build prompt with context
                prompt = self.prompt_template.build_rag_prompt(
                    question=query_request.question,
                    context_chunks=search_results
                )
                
                # Generate answer using LLM
                llm_response = await llm_service.generate_response(
                    prompt=prompt,
                    max_tokens=query_request.options.get("max_tokens", 1000),
                    temperature=query_request.options.get("temperature", 0.7)
                )
                
                answer = llm_response["content"]
                
                # Calculate confidence based on source relevance
                avg_score = sum(chunk["score"] for chunk in search_results) / len(search_results)
                confidence_score = min(avg_score * 1.2, 1.0)  # Boost slightly, cap at 1.0
                
                # Format sources
                sources = [
                    SourceReference(
                        chunk_id=uuid.UUID(chunk["chunk_id"]),
                        document_id=uuid.UUID(chunk["document_id"]),
                        document_name=chunk["document_name"],
                        content=chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"],
                        score=chunk["score"],
                        page_number=chunk.get("page_number"),
                        section_title=chunk.get("section_title"),
                        metadata=chunk.get("metadata", {})
                    )
                    for chunk in search_results
                ]
                
                # Update query with LLM details
                query.llm_provider = llm_response["provider"]
                query.llm_model = llm_response.get("model", "unknown")
                query.token_count_input = llm_response.get("usage", {}).get("prompt_tokens", 0)
                query.token_count_output = llm_response.get("usage", {}).get("completion_tokens", 0)
            
            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update query record
            query.answer = answer
            query.confidence_score = confidence_score
            query.response_time_ms = response_time_ms
            query.status = "completed"
            query.sources_used = {
                "chunks": [
                    {
                        "chunk_id": str(source.chunk_id),
                        "score": source.score
                    }
                    for source in sources
                ]
            }
            
            await db.commit()
            
            # Update session stats
            session.total_queries += 1
            session.total_tokens_used += query.token_count_input + query.token_count_output
            await db.commit()
            
            # Log query execution
            logger.log_query_executed(
                query_id=str(query.id),
                question=query_request.question,
                user_id=str(user.id),
                response_time_ms=response_time_ms,
                confidence_score=confidence_score,
                sources_count=len(sources),
                llm_provider=query.llm_provider or "none",
                tokens_used=query.token_count_input + query.token_count_output
            )
            
            return QueryResponse(
                query_id=query.id,
                question=query_request.question,
                answer=answer,
                confidence_score=confidence_score,
                sources=sources,
                response_time_ms=response_time_ms,
                token_count_input=query.token_count_input or 0,
                token_count_output=query.token_count_output or 0,
                llm_provider=query.llm_provider or "none",
                llm_model=query.llm_model or "unknown",
                filters_applied=query_request.filters,
                query_metadata=query_request.options,
                created_at=query.created_at
            )
        
        except Exception as e:
            # Update query with error
            if 'query' in locals():
                query.status = "failed"
                query.error_message = str(e)
                await db.commit()
            
            # Re-raise the exception
            if isinstance(e, (LLMError, VectorStoreError)):
                raise e
            else:
                raise LLMError(f"Query processing failed: {str(e)}")
    
    async def get_query_history(
        self,
        db: AsyncSession,
        user: User,
        session_id: Optional[uuid.UUID] = None,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[Query], int]:
        """Get query history for user."""
        stmt = (
            select(Query)
            .join(QuerySession)
            .where(QuerySession.user_id == user.id)
        )
        
        if session_id:
            stmt = stmt.where(Query.session_id == session_id)
        
        # Get total count
        count_stmt = select(func.count(Query.id)).select_from(stmt.subquery())
        count_result = await db.execute(count_stmt)
        total = count_result.scalar()
        
        # Apply pagination and ordering
        stmt = (
            stmt
            .order_by(Query.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(stmt)
        queries = result.scalars().all()
        
        return list(queries), total
    
    async def get_query_by_id(
        self,
        db: AsyncSession,
        query_id: uuid.UUID,
        user: User
    ) -> Optional[Query]:
        """Get query by ID."""
        stmt = (
            select(Query)
            .join(QuerySession)
            .where(Query.id == query_id)
            .where(QuerySession.user_id == user.id)
        )
        
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update_query_feedback(
        self,
        db: AsyncSession,
        query_id: uuid.UUID,
        user: User,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """Update query with user feedback."""
        query = await self.get_query_by_id(db, query_id, user)
        if not query:
            return False
        
        query.user_feedback_rating = rating
        if feedback_text:
            if not query.query_metadata:
                query.query_metadata = {}
            query.query_metadata["feedback"] = feedback_text
        
        await db.commit()
        return True
    
    async def get_session_by_id(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        user: User
    ) -> Optional[QuerySession]:
        """Get query session by ID."""
        stmt = (
            select(QuerySession)
            .where(QuerySession.id == session_id)
            .where(QuerySession.user_id == user.id)
        )
        
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def end_session(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        user: User
    ) -> bool:
        """End a query session."""
        session = await self.get_session_by_id(db, session_id, user)
        if not session or session.session_end:
            return False
        
        session.session_end = datetime.utcnow()
        await db.commit()
        return True
    
    async def get_user_sessions(
        self,
        db: AsyncSession,
        user: User,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[QuerySession], int]:
        """Get user's query sessions."""
        stmt = select(QuerySession).where(QuerySession.user_id == user.id)
        
        # Get total count
        count_stmt = select(func.count(QuerySession.id)).select_from(stmt.subquery())
        count_result = await db.execute(count_stmt)
        total = count_result.scalar()
        
        # Apply pagination and ordering
        stmt = (
            stmt
            .order_by(QuerySession.session_start.desc())
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        return list(sessions), total
    
    async def get_query_suggestions(
        self,
        db: AsyncSession,
        user: User,
        context: Optional[str] = None,
        limit: int = 5
    ) -> List[str]:
        """Get query suggestions based on user history and context."""
        # This is a simplified implementation
        # In practice, you might use ML models or more sophisticated logic
        
        # Get recent successful queries
        stmt = (
            select(Query.question)
            .join(QuerySession)
            .where(QuerySession.user_id == user.id)
            .where(Query.status == "completed")
            .where(Query.confidence_score > 0.7)
            .order_by(Query.created_at.desc())
            .limit(20)
        )
        
        result = await db.execute(stmt)
        recent_questions = [row[0] for row in result.all()]
        
        # Simple suggestion logic - return variations of successful queries
        suggestions = []
        for question in recent_questions[:limit]:
            if len(question.split()) > 3:  # Only suggest longer questions
                suggestions.append(question)
        
        return suggestions
    
    async def get_similar_questions(
        self,
        db: AsyncSession,
        question: str,
        user: User,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar questions from user's history."""
        try:
            # Generate embedding for the input question
            question_embedding = await llm_service.generate_embedding(text=question)
            
            # Get user's query history
            stmt = (
                select(Query)
                .join(QuerySession)
                .where(QuerySession.user_id == user.id)
                .where(Query.status == "completed")
                .order_by(Query.created_at.desc())
                .limit(100)  # Check last 100 queries
            )
            
            result = await db.execute(stmt)
            queries = result.scalars().all()
            
            # Calculate similarity (simplified - in practice you'd use vector similarity)
            similar_questions = []
            for query in queries:
                # Simple text similarity based on word overlap
                question_words = set(question.lower().split())
                query_words = set(query.question.lower().split())
                overlap = len(question_words & query_words)
                total_words = len(question_words | query_words)
                
                if total_words > 0:
                    similarity = overlap / total_words
                    if similarity > 0.3:  # Threshold for similarity
                        similar_questions.append({
                            "question": query.question,
                            "similarity": similarity,
                            "confidence_score": query.confidence_score,
                            "created_at": query.created_at
                        })
            
            # Sort by similarity and return top results
            similar_questions.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_questions[:limit]
        
        except Exception as e:
            # Return empty list if similarity search fails
            return []


# Global query service instance
query_service = QueryService()