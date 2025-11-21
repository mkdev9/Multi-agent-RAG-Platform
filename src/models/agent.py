"""Agent execution models."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import String, Integer, Text, ForeignKey, DateTime, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class AgentExecution(Base):
    """Agent execution model for tracking multi-agent workflows."""
    
    __tablename__ = "agent_executions"
    
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )
    
    # Workflow information
    workflow_name: Mapped[str] = mapped_column(String(100), nullable=False)
    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)  # research, analysis, synthesis
    
    # Execution status
    status: Mapped[str] = mapped_column(
        String(50),
        default="initiated",
        nullable=False
    )  # initiated, planning, executing, completed, failed, cancelled
    
    # Agent information
    agents_used: Mapped[List[str]] = mapped_column(ARRAY(String))  # List of agent types
    current_agent: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Progress tracking
    total_steps: Mapped[Optional[int]] = mapped_column(Integer)
    completed_steps: Mapped[int] = mapped_column(Integer, default=0)
    current_step_description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timing
    execution_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    execution_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    estimated_completion: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    
    # Results
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    intermediate_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Resource usage
    total_tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    total_api_calls: Mapped[Optional[int]] = mapped_column(Integer)
    execution_cost: Mapped[Optional[float]] = mapped_column()  # In USD
    
    # Configuration and parameters
    workflow_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    input_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Quality metrics
    quality_score: Mapped[Optional[float]] = mapped_column()  # 0.0 to 1.0
    user_satisfaction: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 rating
    
    # Relationships and dependencies
    parent_execution_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_executions.id")
    )
    
    # Session tracking
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_sessions.id")
    )
    
    def __repr__(self) -> str:
        return f"<AgentExecution(workflow={self.workflow_name}, status={self.status})>"


class AgentStep(Base):
    """Individual step within an agent execution."""
    
    __tablename__ = "agent_steps"
    
    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_executions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    step_description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Step execution
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False
    )  # pending, running, completed, failed, skipped
    
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Input/Output
    input_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    output_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Tools and resources used
    tools_used: Mapped[Optional[List[str]]] = mapped_column(JSONB)
    api_calls_made: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Error details
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    def __repr__(self) -> str:
        return f"<AgentStep(execution_id={self.execution_id}, step={self.step_number})>"