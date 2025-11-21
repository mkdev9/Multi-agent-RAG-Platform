"""API middleware for logging, rate limiting, and error handling."""
import time
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.exceptions import BaseAPIException
from ..core.logging import RequestLogger, SecurityLogger
from ..config import settings


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Get user info if available
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = str(request.state.user.id)
        
        # Log request
        start_time = time.time()
        RequestLogger.log_request(
            method=request.method,
            path=str(request.url.path),
            user_id=user_id,
            request_id=request_id,
            extra_data={
                "query_params": dict(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log response
        RequestLogger.log_response(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            user_id=user_id,
            request_id=request_id
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling exceptions and returning appropriate responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        
        except BaseAPIException as e:
            # Handle custom API exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.message,
                    "error_code": e.error_code,
                    "details": e.details,
                    "timestamp": time.time()
                }
            )
        
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "error_code": "HTTP_EXCEPTION",
                    "timestamp": time.time()
                }
            )
        
        except Exception as e:
            # Handle unexpected exceptions
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "details": {"message": str(e)} if settings.debug else {},
                    "timestamp": time.time()
                }
            )


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times = {}  # client_ip -> list of request times
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.api.rate_limit_enabled:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        if client_ip in self.request_times:
            self.request_times[client_ip] = [
                req_time for req_time in self.request_times[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.request_times[client_ip] = []
        
        # Check rate limit
        if len(self.request_times[client_ip]) >= self.requests_per_minute:
            SecurityLogger.log_rate_limit_exceeded(
                ip_address=client_ip,
                path=str(request.url.path)
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "timestamp": current_time
                }
            )
        
        # Add current request time
        self.request_times[client_ip].append(current_time)
        
        return await call_next(request)


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        response = await call_next(request)
        
        # Add CORS headers to response
        origin = request.headers.get("origin")
        if origin in settings.api.cors_origins or "*" in settings.api.cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
        
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


# Rate limiting decorator for specific endpoints
def rate_limit(requests_per_minute: int = 60):
    """Decorator for endpoint-specific rate limiting."""
    def decorator(func):
        # In a production system, you'd implement Redis-based rate limiting
        # This is a placeholder for the decorator pattern
        return func
    return decorator