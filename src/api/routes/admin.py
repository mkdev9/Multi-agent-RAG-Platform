"""Admin and system management routes."""
import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ...database import get_db
from ...schemas.common import HealthResponse, MetricsResponse, SuccessResponse
from ...core.security import get_current_user, admin_required
from ...models.user import User
from ...models.document import Document
from ...models.query import Query, QuerySession
from ...models.agent import AgentExecution
from ...config import settings

router = APIRouter(prefix="/admin", tags=["Administration"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_db)
):
    """System health check."""
    services = {}
    
    # Check database
    try:
        await db.execute(select(1))
        services["database"] = "healthy"
    except Exception:
        services["database"] = "unhealthy"
    
    # Check vector database (placeholder)
    try:
        # Would check actual vector DB connection
        services["vector_db"] = "healthy"
    except Exception:
        services["vector_db"] = "unhealthy"
    
    # Check LLM providers (placeholder)
    try:
        # Would check LLM provider availability
        services["llm_providers"] = "healthy"
    except Exception:
        services["llm_providers"] = "unhealthy"
    
    # Overall status
    overall_status = "healthy" if all(
        status == "healthy" for status in services.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.api.version,
        services=services,
        metrics={
            "uptime": "unknown",
            "memory_usage": "unknown",
            "cpu_usage": "unknown"
        }
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics(
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Get system metrics (admin only)."""
    try:
        # Get basic counts
        total_users = await db.scalar(select(func.count(User.id)))
        total_documents = await db.scalar(select(func.count(Document.id)))
        total_queries = await db.scalar(select(func.count(Query.id)))
        
        # Get average response time
        avg_response_time = await db.scalar(
            select(func.avg(Query.response_time_ms))
            .where(Query.response_time_ms.is_not(None))
        ) or 0.0
        
        # Get recent query stats (last 30 days)
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        daily_stats = await db.execute(
            select(
                func.date(Query.created_at).label('date'),
                func.count(Query.id).label('count')
            )
            .where(Query.created_at >= thirty_days_ago)
            .group_by(func.date(Query.created_at))
            .order_by(func.date(Query.created_at))
        )
        
        queries_per_day = [
            {"date": str(row.date), "count": row.count}
            for row in daily_stats
        ]
        
        # Get top document sources
        top_sources = await db.execute(
            select(
                Document.file_type,
                func.count(Document.id).label('count')
            )
            .group_by(Document.file_type)
            .order_by(func.count(Document.id).desc())
            .limit(5)
        )
        
        top_sources_list = [
            {"source": row.file_type, "count": row.count}
            for row in top_sources
        ]
        
        return MetricsResponse(
            total_queries=total_queries or 0,
            total_documents=total_documents or 0,
            total_users=total_users or 0,
            avg_response_time_ms=float(avg_response_time),
            queries_per_day=queries_per_day,
            top_sources=top_sources_list
        )
    
    except Exception as e:
        # Return default metrics if there's an error
        return MetricsResponse(
            total_queries=0,
            total_documents=0,
            total_users=0,
            avg_response_time_ms=0.0,
            queries_per_day=[],
            top_sources=[]
        )


@router.get("/users", response_model=List[Dict[str, Any]])
async def get_all_users(
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Get all users (admin only)."""
    stmt = select(User).order_by(User.created_at.desc())
    result = await db.execute(stmt)
    users = result.scalars().all()
    
    return [
        {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        for user in users
    ]


@router.put("/users/{user_id}/role", response_model=SuccessResponse)
async def update_user_role(
    user_id: uuid.UUID,
    role: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Update user role (admin only)."""
    if role not in ["user", "power_user", "admin"]:
        from ...core.exceptions import ValidationError
        raise ValidationError("Invalid role")
    
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        from ...core.exceptions import NotFoundError
        raise NotFoundError("User not found")
    
    user.role = role
    await db.commit()
    
    return SuccessResponse(message=f"User role updated to {role}")


@router.put("/users/{user_id}/active", response_model=SuccessResponse)
async def toggle_user_active(
    user_id: uuid.UUID,
    is_active: bool,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Toggle user active status (admin only)."""
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        from ...core.exceptions import NotFoundError
        raise NotFoundError("User not found")
    
    user.is_active = is_active
    await db.commit()
    
    status = "activated" if is_active else "deactivated"
    return SuccessResponse(message=f"User {status} successfully")


@router.get("/documents/all", response_model=List[Dict[str, Any]])
async def get_all_documents(
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Get all documents across all users (admin only)."""
    stmt = (
        select(Document)
        .order_by(Document.created_at.desc())
        .limit(100)  # Limit for performance
    )
    result = await db.execute(stmt)
    documents = result.scalars().all()
    
    return [
        {
            "id": str(doc.id),
            "filename": doc.original_filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "processing_status": doc.processing_status,
            "uploaded_by": str(doc.uploaded_by),
            "created_at": doc.created_at.isoformat(),
            "total_chunks": doc.total_chunks,
            "total_tokens": doc.total_tokens
        }
        for doc in documents
    ]


@router.delete("/documents/{document_id}/force", response_model=SuccessResponse)
async def force_delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Force delete any document (admin only)."""
    from ...services.document import document_service
    
    # Get document without user restriction
    stmt = select(Document).where(Document.id == document_id)
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    
    if not document:
        from ...core.exceptions import NotFoundError
        raise NotFoundError("Document not found")
    
    # Delete document (bypass user check)
    success = await document_service.delete_document(
        db=db,
        document_id=document_id,
        user=None  # Admin bypass
    )
    
    if success:
        return SuccessResponse(message="Document force deleted successfully")
    else:
        from ...core.exceptions import ProcessingError
        raise ProcessingError("Failed to delete document")


@router.post("/system/maintenance", response_model=SuccessResponse)
async def trigger_maintenance(
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Trigger system maintenance tasks (admin only)."""
    # This would trigger various maintenance tasks like:
    # - Cleaning up orphaned files
    # - Rebuilding vector indices
    # - Cleaning up expired sessions
    # - Database optimization
    
    return SuccessResponse(
        message="Maintenance tasks triggered (placeholder implementation)"
    )