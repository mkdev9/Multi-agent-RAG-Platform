"""Custom exceptions for the application."""


class BaseAPIException(Exception):
    """Base exception for API errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = None,
        details: dict = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(BaseAPIException):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication failed", details: dict = None):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationError(BaseAPIException):
    """Authorization error."""
    
    def __init__(self, message: str = "Insufficient permissions", details: dict = None):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            details=details
        )


class ValidationError(BaseAPIException):
    """Validation error."""
    
    def __init__(self, message: str = "Validation failed", details: dict = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(BaseAPIException):
    """Resource not found error."""
    
    def __init__(self, message: str = "Resource not found", details: dict = None):
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND_ERROR",
            details=details
        )


class ConflictError(BaseAPIException):
    """Resource conflict error."""
    
    def __init__(self, message: str = "Resource conflict", details: dict = None):
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT_ERROR",
            details=details
        )


class ProcessingError(BaseAPIException):
    """Document processing error."""
    
    def __init__(self, message: str = "Processing failed", details: dict = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="PROCESSING_ERROR",
            details=details
        )


class VectorStoreError(BaseAPIException):
    """Vector store operation error."""
    
    def __init__(self, message: str = "Vector store operation failed", details: dict = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="VECTOR_STORE_ERROR",
            details=details
        )


class LLMError(BaseAPIException):
    """LLM service error."""
    
    def __init__(self, message: str = "LLM service error", details: dict = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="LLM_ERROR",
            details=details
        )


class AgentExecutionError(BaseAPIException):
    """Agent execution error."""
    
    def __init__(self, message: str = "Agent execution failed", details: dict = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="AGENT_EXECUTION_ERROR",
            details=details
        )


class RateLimitExceeded(BaseAPIException):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: dict = None):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class ExternalServiceError(BaseAPIException):
    """External service error."""
    
    def __init__(self, message: str = "External service error", details: dict = None):
        super().__init__(
            message=message,
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details
        )


class ConfigurationError(BaseAPIException):
    """Configuration error."""
    
    def __init__(self, message: str = "Configuration error", details: dict = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details=details
        )