"""Agent execution schemas."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from .common import BaseSchema


class AgentExecutionRequest(BaseSchema):
    """Agent execution request schema."""
    
    task: str = Field(..., description="Task description")
    workflow: str = Field(..., description="Workflow type")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Workflow parameters"
    )
    agents: Optional[List[str]] = Field(
        None,
        description="Specific agents to use"
    )
    priority: str = Field(
        default="normal",
        description="Task priority (low, normal, high)"
    )
    
    @validator('workflow')
    def validate_workflow(cls, v):
        """Validate workflow type."""
        allowed_workflows = {
            'research',
            'analysis', 
            'summarization',
            'content_generation',
            'fact_checking',
            'multi_document_qa',
            'comparative_analysis'
        }
        
        if v not in allowed_workflows:
            raise ValueError(f"Invalid workflow: {v}. Allowed: {allowed_workflows}")
        
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority."""
        if v not in ['low', 'normal', 'high']:
            raise ValueError("Priority must be: low, normal, or high")
        return v


class AgentStepResponse(BaseSchema):
    """Agent step response schema."""
    
    id: uuid.UUID = Field(..., description="Step ID")
    step_number: int = Field(..., description="Step number")
    agent_name: str = Field(..., description="Agent name")
    step_description: str = Field(..., description="Step description")
    status: str = Field(..., description="Step status")
    
    start_time: Optional[datetime] = Field(None, description="Step start time")
    end_time: Optional[datetime] = Field(None, description="Step end time")
    duration_ms: Optional[int] = Field(None, description="Step duration")
    
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    
    tools_used: Optional[List[str]] = Field(None, description="Tools used")
    api_calls_made: Optional[int] = Field(None, description="API calls made")
    tokens_used: Optional[int] = Field(None, description="Tokens used")
    
    error_message: Optional[str] = Field(None, description="Error message")
    retry_count: int = Field(..., description="Retry count")


class AgentExecutionResponse(BaseSchema):
    """Agent execution response schema."""
    
    id: uuid.UUID = Field(..., description="Execution ID")
    user_id: uuid.UUID = Field(..., description="User ID")
    workflow_name: str = Field(..., description="Workflow name")
    task_description: str = Field(..., description="Task description")
    task_type: str = Field(..., description="Task type")
    
    status: str = Field(..., description="Execution status")
    agents_used: List[str] = Field(..., description="Agents used")
    current_agent: Optional[str] = Field(None, description="Current agent")
    
    # Progress
    total_steps: Optional[int] = Field(None, description="Total steps")
    completed_steps: int = Field(..., description="Completed steps")
    current_step_description: Optional[str] = Field(None, description="Current step")
    progress_percentage: Optional[float] = Field(None, description="Progress percentage")
    
    # Timing
    execution_start: datetime = Field(..., description="Execution start")
    execution_end: Optional[datetime] = Field(None, description="Execution end")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")
    duration_ms: Optional[int] = Field(None, description="Execution duration")
    
    # Results
    result: Optional[Dict[str, Any]] = Field(None, description="Final result")
    intermediate_results: Optional[Dict[str, Any]] = Field(None, description="Intermediate results")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message")
    retry_count: int = Field(..., description="Retry count")
    
    # Resource usage
    total_tokens_used: Optional[int] = Field(None, description="Total tokens used")
    total_api_calls: Optional[int] = Field(None, description="Total API calls")
    execution_cost: Optional[float] = Field(None, description="Execution cost (USD)")
    
    # Quality metrics
    quality_score: Optional[float] = Field(None, description="Quality score")
    user_satisfaction: Optional[int] = Field(None, description="User satisfaction")
    
    # Configuration
    workflow_config: Optional[Dict[str, Any]] = Field(None, description="Workflow config")
    input_parameters: Optional[Dict[str, Any]] = Field(None, description="Input parameters")
    
    # Steps (optional, for detailed view)
    steps: Optional[List[AgentStepResponse]] = Field(None, description="Execution steps")
    
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")


class WorkflowDefinition(BaseSchema):
    """Workflow definition schema."""
    
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    agents: List[str] = Field(..., description="Required agents")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Default parameters"
    )
    estimated_duration_ms: Optional[int] = Field(
        None,
        description="Estimated duration"
    )


class AgentDefinition(BaseSchema):
    """Agent definition schema."""
    
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    tools: List[str] = Field(..., description="Available tools")
    model_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Model configuration"
    )
    

class ExecutionFeedback(BaseSchema):
    """Execution feedback schema."""
    
    execution_id: uuid.UUID = Field(..., description="Execution ID")
    satisfaction: int = Field(..., ge=1, le=5, description="Satisfaction rating")
    feedback_text: Optional[str] = Field(None, description="Feedback text")
    quality_rating: Optional[int] = Field(
        None, ge=1, le=5, description="Quality rating"
    )
    speed_rating: Optional[int] = Field(
        None, ge=1, le=5, description="Speed rating"
    )
    accuracy_rating: Optional[int] = Field(
        None, ge=1, le=5, description="Accuracy rating"
    )
    suggestions: Optional[str] = Field(None, description="Improvement suggestions")


class AgentMetrics(BaseSchema):
    """Agent metrics schema."""
    
    agent_name: str = Field(..., description="Agent name")
    total_executions: int = Field(..., description="Total executions")
    success_rate: float = Field(..., description="Success rate")
    avg_execution_time_ms: float = Field(..., description="Average execution time")
    avg_quality_score: Optional[float] = Field(None, description="Average quality score")
    most_common_tasks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most common tasks"
    )


class SystemMetrics(BaseSchema):
    """System-wide agent metrics."""
    
    total_executions: int = Field(..., description="Total executions")
    active_executions: int = Field(..., description="Active executions")
    avg_queue_time_ms: float = Field(..., description="Average queue time")
    system_load: float = Field(..., description="System load")
    agent_metrics: List[AgentMetrics] = Field(
        default_factory=list,
        description="Per-agent metrics"
    )