"""Agent execution routes."""
import uuid
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...schemas.agent import (
    AgentExecutionRequest, AgentExecutionResponse,
    ExecutionFeedback, AgentMetrics, SystemMetrics
)
from ...schemas.common import PaginatedResponse, SuccessResponse, PaginationParams
from ...core.security import get_current_user
from ...core.exceptions import NotFoundError
from ...services.agent import agent_service
from ...models.user import User

router = APIRouter(prefix="/agents", tags=["Agent Orchestration"])


@router.post("/run", response_model=AgentExecutionResponse)
async def run_agent_workflow(
    request: AgentExecutionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Execute an agent workflow."""
    execution = await agent_service.execute_agent_workflow(
        db=db,
        user=current_user,
        request=request
    )
    
    return AgentExecutionResponse.model_validate(execution)


@router.get("/executions", response_model=PaginatedResponse[AgentExecutionResponse])
async def get_agent_executions(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's agent executions."""
    executions = await agent_service.get_user_executions(
        db=db,
        user=current_user,
        skip=pagination.offset,
        limit=pagination.size
    )
    
    # Convert to response models
    execution_responses = [
        AgentExecutionResponse.model_validate(execution)
        for execution in executions
    ]
    
    return PaginatedResponse.create(
        items=execution_responses,
        total=len(execution_responses),
        page=pagination.page,
        size=pagination.size
    )


@router.get("/executions/{execution_id}", response_model=AgentExecutionResponse)
async def get_agent_execution(
    execution_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific agent execution."""
    execution = await agent_service.get_execution_by_id(
        db=db,
        execution_id=execution_id,
        user=current_user
    )
    
    if not execution:
        raise NotFoundError("Agent execution not found")
    
    return AgentExecutionResponse.model_validate(execution)


@router.post("/executions/{execution_id}/cancel", response_model=SuccessResponse)
async def cancel_agent_execution(
    execution_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Cancel an ongoing agent execution."""
    success = await agent_service.cancel_execution(
        db=db,
        execution_id=execution_id,
        user=current_user
    )
    
    if not success:
        raise NotFoundError("Execution not found or cannot be cancelled")
    
    return SuccessResponse(message="Execution cancelled successfully")


@router.post("/executions/{execution_id}/feedback", response_model=SuccessResponse)
async def submit_execution_feedback(
    execution_id: uuid.UUID,
    feedback: ExecutionFeedback,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback for an agent execution."""
    execution = await agent_service.get_execution_by_id(
        db=db,
        execution_id=execution_id,
        user=current_user
    )
    
    if not execution:
        raise NotFoundError("Execution not found")
    
    # Update execution with feedback
    execution.user_satisfaction = feedback.satisfaction
    execution.quality_score = feedback.quality_rating
    
    # Store detailed feedback in metadata
    if not execution.workflow_config:
        execution.workflow_config = {}
    
    execution.workflow_config.update({
        "user_feedback": {
            "satisfaction": feedback.satisfaction,
            "feedback_text": feedback.feedback_text,
            "quality_rating": feedback.quality_rating,
            "speed_rating": feedback.speed_rating,
            "accuracy_rating": feedback.accuracy_rating,
            "suggestions": feedback.suggestions,
        }
    })
    
    await db.commit()
    
    return SuccessResponse(message="Feedback submitted successfully")


@router.get("/workflows", response_model=List[dict])
async def get_available_workflows():
    """Get list of available workflows."""
    workflows = [
        {
            "name": "research",
            "description": "Comprehensive research using knowledge base and external sources",
            "agents": ["planning_agent", "research_agent", "synthesis_agent", "validation_agent"],
            "estimated_duration": "5-10 minutes",
            "use_cases": ["Market research", "Technical analysis", "Academic research"]
        },
        {
            "name": "analysis",
            "description": "Analyze information and provide insights",
            "agents": ["research_agent", "synthesis_agent"],
            "estimated_duration": "3-5 minutes",
            "use_cases": ["Data analysis", "Comparative analysis", "Trend analysis"]
        },
        {
            "name": "summarization",
            "description": "Create concise summaries of information",
            "agents": ["research_agent", "synthesis_agent"],
            "estimated_duration": "2-3 minutes",
            "use_cases": ["Document summarization", "Report creation", "Executive briefings"]
        },
        {
            "name": "content_generation",
            "description": "Generate content based on research and requirements",
            "agents": ["research_agent", "synthesis_agent"],
            "estimated_duration": "5-8 minutes",
            "use_cases": ["Article writing", "Report generation", "Content creation"]
        }
    ]
    
    return workflows


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get system-wide agent metrics (placeholder implementation)."""
    # This would be implemented with actual metrics collection
    return SystemMetrics(
        total_executions=0,
        active_executions=0,
        avg_queue_time_ms=0.0,
        system_load=0.0,
        agent_metrics=[]
    )