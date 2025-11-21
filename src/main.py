"""Main FastAPI application."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import settings
from .core.logging import configure_logging
from .database import init_db, close_db
from .api.middleware import (
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
    RateLimitingMiddleware,
    SecurityHeadersMiddleware
)
from .api.routes import auth, documents, query, agents, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    configure_logging()
    await init_db()
    yield
    # Shutdown
    await close_db()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.api.title,
        description=settings.api.description,
        version=settings.api.version,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        RateLimitingMiddleware,
        requests_per_minute=settings.api.rate_limit_requests
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(agents.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "RAG Knowledge Platform API",
            "version": settings.api.version,
            "status": "healthy"
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers if not settings.api.reload else 1,
    )