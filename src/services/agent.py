"""Agent orchestration service using LangChain framework."""
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from langchain.agents import Tool
from langchain.agents.agent import AgentExecutor
from langchain.agents.conversational.base import ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool

from ..config import settings
from ..models.user import User
from ..models.agent import AgentExecution, AgentStep
from ..schemas.agent import AgentExecutionRequest, AgentExecutionResponse
from ..core.exceptions import AgentExecutionError, ConfigurationError
from ..core.logging import BusinessLogger
from .llm import llm_service
from .query import query_service
from .vector_store import vector_store_service

logger = BusinessLogger()


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools = []
    
    @abstractmethod
    async def execute_task(
        self,
        task: str,
        context: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute agent task."""
        pass
    
    def get_tools(self) -> List[Tool]:
        """Get available tools for this agent."""
        return self.tools


class RAGSearchTool(BaseTool):
    """Tool for searching knowledge base using RAG."""
    
    name = "knowledge_search"
    description = "Search the knowledge base for information. Input should be a clear question or search query."
    
    def __init__(self, user: User, db: AsyncSession):
        super().__init__()
        self.user = user
        self.db = db
    
    async def _arun(self, query: str) -> str:
        """Async implementation."""
        try:
            from ..schemas.query import QueryRequest
            
            # Create query request
            query_request = QueryRequest(
                question=query,
                options={"max_chunks": 5, "similarity_threshold": 0.7}
            )
            
            # Execute query
            response = await query_service.process_query(
                db=self.db,
                user=self.user,
                query_request=query_request
            )
            
            # Format response for agent
            result = f"Answer: {response.answer}\n\n"
            if response.sources:
                result += "Sources:\n"
                for i, source in enumerate(response.sources[:3], 1):
                    result += f"{i}. {source.document_name} - {source.content[:200]}...\n"
            
            return result
        
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    def _run(self, query: str) -> str:
        """Sync implementation (not used in async context)."""
        return "Sync execution not supported for RAG search"


class WebSearchTool(BaseTool):
    """Tool for searching the web (placeholder implementation)."""
    
    name = "web_search"
    description = "Search the web for current information. Input should be a search query."
    
    async def _arun(self, query: str) -> str:
        """Async implementation."""
        # Placeholder - would integrate with actual web search API
        return f"Web search results for '{query}': [This is a placeholder for web search results]"
    
    def _run(self, query: str) -> str:
        """Sync implementation."""
        return "Sync execution not supported for web search"


class ResearchAgent(BaseAgent):
    """Agent specialized in research tasks."""
    
    def __init__(self, user: User, db: AsyncSession):
        super().__init__(
            name="research_agent",
            description="Conducts research using knowledge base and external sources"
        )
        self.user = user
        self.db = db
        self.tools = [
            RAGSearchTool(user, db),
            WebSearchTool(),
        ]
    
    async def execute_task(
        self,
        task: str,
        context: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute research task."""
        try:
            # Extract research topics from task
            research_queries = self._extract_research_queries(task)
            
            results = []
            for query in research_queries:
                # Search knowledge base
                rag_tool = RAGSearchTool(self.user, db)
                rag_result = await rag_tool._arun(query)
                
                results.append({
                    "query": query,
                    "knowledge_base_result": rag_result,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return {
                "agent": self.name,
                "task_completed": True,
                "research_results": results,
                "summary": f"Completed research on {len(research_queries)} topics"
            }
        
        except Exception as e:
            return {
                "agent": self.name,
                "task_completed": False,
                "error": str(e)
            }
    
    def _extract_research_queries(self, task: str) -> List[str]:
        """Extract research queries from task description."""
        # Simple implementation - in practice would use NLP
        # to intelligently extract subtopics
        if "about" in task.lower():
            # Extract main topic
            topic = task.lower().split("about")[-1].strip()
            return [f"What is {topic}?", f"Key facts about {topic}"]
        else:
            return [task]


class PlanningAgent(BaseAgent):
    """Agent specialized in task planning and decomposition."""
    
    def __init__(self):
        super().__init__(
            name="planning_agent",
            description="Breaks down complex tasks into actionable steps"
        )
    
    async def execute_task(
        self,
        task: str,
        context: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute planning task."""
        try:
            # Use LLM to break down task
            planning_prompt = f"""
            Task: {task}
            
            Please break down this task into a series of logical steps. 
            Each step should be clear and actionable.
            Format your response as a numbered list.
            """
            
            llm_response = await llm_service.generate_response(
                prompt=planning_prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse steps (simple implementation)
            steps_text = llm_response["content"]
            steps = self._parse_steps(steps_text)
            
            return {
                "agent": self.name,
                "task_completed": True,
                "execution_plan": steps,
                "total_steps": len(steps),
                "estimated_duration": len(steps) * 2  # 2 minutes per step estimate
            }
        
        except Exception as e:
            return {
                "agent": self.name,
                "task_completed": False,
                "error": str(e)
            }
    
    def _parse_steps(self, steps_text: str) -> List[Dict[str, Any]]:
        """Parse steps from LLM response."""
        steps = []
        lines = steps_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                # Remove numbering/bullets
                step_text = line.lstrip('0123456789.•- ').strip()
                if step_text:
                    steps.append({
                        "step_number": len(steps) + 1,
                        "description": step_text,
                        "estimated_duration": 120  # 2 minutes default
                    })
        
        return steps


class SynthesisAgent(BaseAgent):
    """Agent specialized in synthesizing information."""
    
    def __init__(self):
        super().__init__(
            name="synthesis_agent",
            description="Combines and synthesizes information from multiple sources"
        )
    
    async def execute_task(
        self,
        task: str,
        context: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute synthesis task."""
        try:
            # Get information from context
            research_results = context.get("research_results", [])
            
            if not research_results:
                return {
                    "agent": self.name,
                    "task_completed": False,
                    "error": "No research results to synthesize"
                }
            
            # Combine all research findings
            combined_content = []
            for result in research_results:
                if "knowledge_base_result" in result:
                    combined_content.append(result["knowledge_base_result"])
            
            # Use LLM to synthesize
            synthesis_prompt = f"""
            Task: {task}
            
            Please synthesize the following information into a coherent and comprehensive response:
            
            {' '.join(combined_content)}
            
            Provide a well-structured synthesis that addresses the original task.
            """
            
            llm_response = await llm_service.generate_response(
                prompt=synthesis_prompt,
                max_tokens=1000,
                temperature=0.5
            )
            
            return {
                "agent": self.name,
                "task_completed": True,
                "synthesized_content": llm_response["content"],
                "sources_processed": len(research_results),
                "token_usage": llm_response.get("usage", {})
            }
        
        except Exception as e:
            return {
                "agent": self.name,
                "task_completed": False,
                "error": str(e)
            }


class ValidationAgent(BaseAgent):
    """Agent specialized in validating and fact-checking."""
    
    def __init__(self, user: User, db: AsyncSession):
        super().__init__(
            name="validation_agent",
            description="Validates information accuracy and completeness"
        )
        self.user = user
        self.db = db
    
    async def execute_task(
        self,
        task: str,
        context: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute validation task."""
        try:
            synthesized_content = context.get("synthesized_content", "")
            
            if not synthesized_content:
                return {
                    "agent": self.name,
                    "task_completed": False,
                    "error": "No content to validate"
                }
            
            # Use LLM to validate content
            validation_prompt = f"""
            Please review the following content for accuracy, completeness, and logical consistency:
            
            {synthesized_content}
            
            Provide:
            1. Overall quality score (1-10)
            2. Areas that need improvement
            3. Fact-checking results
            4. Suggestions for enhancement
            """
            
            llm_response = await llm_service.generate_response(
                prompt=validation_prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            # Parse validation results (simple implementation)
            validation_text = llm_response["content"]
            quality_score = self._extract_quality_score(validation_text)
            
            return {
                "agent": self.name,
                "task_completed": True,
                "validation_report": validation_text,
                "quality_score": quality_score,
                "validation_passed": quality_score >= 7.0,
                "recommendations": self._extract_recommendations(validation_text)
            }
        
        except Exception as e:
            return {
                "agent": self.name,
                "task_completed": False,
                "error": str(e)
            }
    
    def _extract_quality_score(self, text: str) -> float:
        """Extract quality score from validation text."""
        # Simple regex to find score
        import re
        score_match = re.search(r'score.*?(\d+(?:\.\d+)?)', text.lower())
        if score_match:
            return float(score_match.group(1))
        return 5.0  # Default score
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from validation text."""
        # Simple implementation
        lines = text.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['suggest', 'recommend', 'improve', 'enhance']):
                recommendations.append(line)
        
        return recommendations


class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows."""
    
    def __init__(self):
        self.workflows = {
            "research": self._research_workflow,
            "analysis": self._analysis_workflow,
            "summarization": self._summarization_workflow,
            "content_generation": self._content_generation_workflow,
        }
    
    async def execute_workflow(
        self,
        workflow_name: str,
        task: str,
        user: User,
        db: AsyncSession,
        execution: AgentExecution,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a workflow."""
        if workflow_name not in self.workflows:
            raise AgentExecutionError(f"Unknown workflow: {workflow_name}")
        
        try:
            # Update execution status
            await self._update_execution_status(db, execution.id, "executing")
            
            # Execute workflow
            workflow_func = self.workflows[workflow_name]
            result = await workflow_func(task, user, db, execution, parameters or {})
            
            # Update execution with results
            await self._update_execution_status(
                db, execution.id, "completed", result=result
            )
            
            return result
        
        except Exception as e:
            await self._update_execution_status(
                db, execution.id, "failed", error_message=str(e)
            )
            raise AgentExecutionError(f"Workflow execution failed: {str(e)}")
    
    async def _research_workflow(
        self,
        task: str,
        user: User,
        db: AsyncSession,
        execution: AgentExecution,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Research workflow implementation."""
        context = {}
        
        # Step 1: Planning
        await self._create_step(db, execution.id, 1, "planning_agent", "Planning research approach")
        planner = PlanningAgent()
        plan_result = await planner.execute_task(task, context, db)
        context.update(plan_result)
        
        # Step 2: Research
        await self._create_step(db, execution.id, 2, "research_agent", "Conducting research")
        researcher = ResearchAgent(user, db)
        research_result = await researcher.execute_task(task, context, db)
        context.update(research_result)
        
        # Step 3: Synthesis
        await self._create_step(db, execution.id, 3, "synthesis_agent", "Synthesizing findings")
        synthesizer = SynthesisAgent()
        synthesis_result = await synthesizer.execute_task(task, context, db)
        context.update(synthesis_result)
        
        # Step 4: Validation
        await self._create_step(db, execution.id, 4, "validation_agent", "Validating results")
        validator = ValidationAgent(user, db)
        validation_result = await validator.execute_task(task, context, db)
        context.update(validation_result)
        
        return {
            "workflow": "research",
            "task": task,
            "final_result": context.get("synthesized_content", ""),
            "quality_score": context.get("quality_score", 0.0),
            "validation_passed": context.get("validation_passed", False),
            "steps_completed": 4,
            "context": context
        }
    
    async def _analysis_workflow(
        self,
        task: str,
        user: User,
        db: AsyncSession,
        execution: AgentExecution,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analysis workflow implementation."""
        # Simplified analysis workflow
        context = {}
        
        # Research step
        await self._create_step(db, execution.id, 1, "research_agent", "Gathering information for analysis")
        researcher = ResearchAgent(user, db)
        research_result = await researcher.execute_task(task, context, db)
        context.update(research_result)
        
        # Analysis step
        await self._create_step(db, execution.id, 2, "synthesis_agent", "Analyzing information")
        analysis_task = f"Analyze the following information and provide insights: {task}"
        synthesizer = SynthesisAgent()
        analysis_result = await synthesizer.execute_task(analysis_task, context, db)
        
        return {
            "workflow": "analysis",
            "task": task,
            "analysis_result": analysis_result.get("synthesized_content", ""),
            "steps_completed": 2
        }
    
    async def _summarization_workflow(
        self,
        task: str,
        user: User,
        db: AsyncSession,
        execution: AgentExecution,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Summarization workflow implementation."""
        context = {}
        
        # Research content
        await self._create_step(db, execution.id, 1, "research_agent", "Gathering content to summarize")
        researcher = ResearchAgent(user, db)
        research_result = await researcher.execute_task(task, context, db)
        context.update(research_result)
        
        # Summarize
        await self._create_step(db, execution.id, 2, "synthesis_agent", "Creating summary")
        summary_task = f"Create a concise summary of: {task}"
        synthesizer = SynthesisAgent()
        summary_result = await synthesizer.execute_task(summary_task, context, db)
        
        return {
            "workflow": "summarization",
            "task": task,
            "summary": summary_result.get("synthesized_content", ""),
            "steps_completed": 2
        }
    
    async def _content_generation_workflow(
        self,
        task: str,
        user: User,
        db: AsyncSession,
        execution: AgentExecution,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Content generation workflow implementation."""
        context = parameters
        
        # Research background information
        await self._create_step(db, execution.id, 1, "research_agent", "Researching background information")
        researcher = ResearchAgent(user, db)
        research_result = await researcher.execute_task(task, context, db)
        context.update(research_result)
        
        # Generate content
        await self._create_step(db, execution.id, 2, "synthesis_agent", "Generating content")
        synthesizer = SynthesisAgent()
        content_result = await synthesizer.execute_task(task, context, db)
        
        return {
            "workflow": "content_generation",
            "task": task,
            "generated_content": content_result.get("synthesized_content", ""),
            "steps_completed": 2
        }
    
    async def _create_step(
        self,
        db: AsyncSession,
        execution_id: uuid.UUID,
        step_number: int,
        agent_name: str,
        description: str
    ):
        """Create an agent step record."""
        step = AgentStep(
            execution_id=execution_id,
            step_number=step_number,
            agent_name=agent_name,
            step_description=description,
            status="running",
            start_time=datetime.utcnow()
        )
        db.add(step)
        await db.commit()
    
    async def _update_execution_status(
        self,
        db: AsyncSession,
        execution_id: uuid.UUID,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Update execution status."""
        update_data = {"status": status}
        
        if result:
            update_data["result"] = result
        
        if error_message:
            update_data["error_message"] = error_message
        
        if status in ["completed", "failed"]:
            update_data["execution_end"] = datetime.utcnow()
        
        await db.execute(
            update(AgentExecution)
            .where(AgentExecution.id == execution_id)
            .values(**update_data)
        )
        await db.commit()


class AgentService:
    """Main agent service for managing agent executions."""
    
    def __init__(self):
        self.orchestrator = WorkflowOrchestrator()
    
    async def execute_agent_workflow(
        self,
        db: AsyncSession,
        user: User,
        request: AgentExecutionRequest
    ) -> AgentExecution:
        """Execute an agent workflow."""
        try:
            # Create execution record
            execution = AgentExecution(
                user_id=user.id,
                workflow_name=request.workflow,
                task_description=request.task,
                task_type=request.workflow,
                status="initiated",
                agents_used=request.agents or [],
                execution_start=datetime.utcnow(),
                workflow_config={"priority": request.priority},
                input_parameters=request.parameters
            )
            
            db.add(execution)
            await db.commit()
            await db.refresh(execution)
            
            # Log execution start
            logger.log_agent_execution_started(
                execution_id=str(execution.id),
                workflow=request.workflow,
                task_description=request.task,
                user_id=str(user.id),
                agents=request.agents or []
            )
            
            # Execute workflow asynchronously
            asyncio.create_task(
                self._execute_workflow_async(
                    db, execution, request.workflow, request.task, user, request.parameters
                )
            )
            
            return execution
        
        except Exception as e:
            raise AgentExecutionError(f"Failed to start agent execution: {str(e)}")
    
    async def _execute_workflow_async(
        self,
        db: AsyncSession,
        execution: AgentExecution,
        workflow: str,
        task: str,
        user: User,
        parameters: Optional[Dict[str, Any]]
    ):
        """Execute workflow asynchronously."""
        start_time = datetime.utcnow()
        
        try:
            result = await self.orchestrator.execute_workflow(
                workflow_name=workflow,
                task=task,
                user=user,
                db=db,
                execution=execution,
                parameters=parameters
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.log_agent_execution_completed(
                execution_id=str(execution.id),
                workflow=workflow,
                success=True,
                execution_time_ms=execution_time
            )
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.log_agent_execution_completed(
                execution_id=str(execution.id),
                workflow=workflow,
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    async def get_execution_by_id(
        self,
        db: AsyncSession,
        execution_id: uuid.UUID,
        user: User
    ) -> Optional[AgentExecution]:
        """Get execution by ID."""
        stmt = (
            select(AgentExecution)
            .where(AgentExecution.id == execution_id)
            .where(AgentExecution.user_id == user.id)
        )
        
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_user_executions(
        self,
        db: AsyncSession,
        user: User,
        skip: int = 0,
        limit: int = 20
    ) -> List[AgentExecution]:
        """Get user's agent executions."""
        stmt = (
            select(AgentExecution)
            .where(AgentExecution.user_id == user.id)
            .order_by(AgentExecution.execution_start.desc())
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(stmt)
        return list(result.scalars().all())
    
    async def cancel_execution(
        self,
        db: AsyncSession,
        execution_id: uuid.UUID,
        user: User
    ) -> bool:
        """Cancel an ongoing execution."""
        execution = await self.get_execution_by_id(db, execution_id, user)
        if not execution or execution.status in ["completed", "failed", "cancelled"]:
            return False
        
        await db.execute(
            update(AgentExecution)
            .where(AgentExecution.id == execution_id)
            .values(
                status="cancelled",
                execution_end=datetime.utcnow()
            )
        )
        await db.commit()
        
        return True


# Global agent service instance
agent_service = AgentService()