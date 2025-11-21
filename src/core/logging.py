"""Logging configuration and utilities."""
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from ..config import settings


def configure_logging():
    """Configure structured logging."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.monitoring.log_level.upper()),
    )
    
    # Set third-party log levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


class RequestLogger:
    """Request logging utility."""
    
    @staticmethod
    def log_request(
        method: str,
        path: str,
        user_id: str = None,
        request_id: str = None,
        extra_data: Dict[str, Any] = None
    ):
        """Log incoming request."""
        logger = structlog.get_logger("api.request")
        logger.info(
            "Request started",
            method=method,
            path=path,
            user_id=user_id,
            request_id=request_id,
            **(extra_data or {})
        )
    
    @staticmethod
    def log_response(
        method: str,
        path: str,
        status_code: int,
        response_time_ms: float,
        user_id: str = None,
        request_id: str = None,
        extra_data: Dict[str, Any] = None
    ):
        """Log response."""
        logger = structlog.get_logger("api.response")
        logger.info(
            "Request completed",
            method=method,
            path=path,
            status_code=status_code,
            response_time_ms=response_time_ms,
            user_id=user_id,
            request_id=request_id,
            **(extra_data or {})
        )


class BusinessLogger:
    """Business event logging utility."""
    
    @staticmethod
    def log_document_uploaded(
        document_id: str,
        filename: str,
        file_size: int,
        user_id: str,
        processing_options: Dict[str, Any] = None
    ):
        """Log document upload."""
        logger = structlog.get_logger("business.document")
        logger.info(
            "Document uploaded",
            event="document_uploaded",
            document_id=document_id,
            filename=filename,
            file_size=file_size,
            user_id=user_id,
            processing_options=processing_options or {}
        )
    
    @staticmethod
    def log_document_processed(
        document_id: str,
        filename: str,
        chunks_created: int,
        processing_time_ms: float,
        success: bool,
        error_message: str = None
    ):
        """Log document processing completion."""
        logger = structlog.get_logger("business.document")
        logger.info(
            "Document processed",
            event="document_processed",
            document_id=document_id,
            filename=filename,
            chunks_created=chunks_created,
            processing_time_ms=processing_time_ms,
            success=success,
            error_message=error_message
        )
    
    @staticmethod
    def log_query_executed(
        query_id: str,
        question: str,
        user_id: str,
        response_time_ms: float,
        confidence_score: float,
        sources_count: int,
        llm_provider: str,
        tokens_used: int
    ):
        """Log query execution."""
        logger = structlog.get_logger("business.query")
        logger.info(
            "Query executed",
            event="query_executed",
            query_id=query_id,
            question=question,
            user_id=user_id,
            response_time_ms=response_time_ms,
            confidence_score=confidence_score,
            sources_count=sources_count,
            llm_provider=llm_provider,
            tokens_used=tokens_used
        )
    
    @staticmethod
    def log_agent_execution_started(
        execution_id: str,
        workflow: str,
        task_description: str,
        user_id: str,
        agents: list[str]
    ):
        """Log agent execution start."""
        logger = structlog.get_logger("business.agent")
        logger.info(
            "Agent execution started",
            event="agent_execution_started",
            execution_id=execution_id,
            workflow=workflow,
            task_description=task_description,
            user_id=user_id,
            agents=agents
        )
    
    @staticmethod
    def log_agent_execution_completed(
        execution_id: str,
        workflow: str,
        success: bool,
        execution_time_ms: float,
        tokens_used: int = None,
        error_message: str = None
    ):
        """Log agent execution completion."""
        logger = structlog.get_logger("business.agent")
        logger.info(
            "Agent execution completed",
            event="agent_execution_completed",
            execution_id=execution_id,
            workflow=workflow,
            success=success,
            execution_time_ms=execution_time_ms,
            tokens_used=tokens_used,
            error_message=error_message
        )


class SecurityLogger:
    """Security event logging utility."""
    
    @staticmethod
    def log_login_attempt(
        email: str,
        success: bool,
        ip_address: str = None,
        user_agent: str = None,
        failure_reason: str = None
    ):
        """Log login attempt."""
        logger = structlog.get_logger("security.auth")
        logger.info(
            "Login attempt",
            event="login_attempt",
            email=email,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            failure_reason=failure_reason
        )
    
    @staticmethod
    def log_unauthorized_access(
        path: str,
        method: str,
        ip_address: str = None,
        user_agent: str = None,
        reason: str = None
    ):
        """Log unauthorized access attempt."""
        logger = structlog.get_logger("security.access")
        logger.warning(
            "Unauthorized access attempt",
            event="unauthorized_access",
            path=path,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            reason=reason
        )
    
    @staticmethod
    def log_rate_limit_exceeded(
        ip_address: str,
        path: str,
        limit_type: str = "general"
    ):
        """Log rate limit exceeded."""
        logger = structlog.get_logger("security.rate_limit")
        logger.warning(
            "Rate limit exceeded",
            event="rate_limit_exceeded",
            ip_address=ip_address,
            path=path,
            limit_type=limit_type
        )