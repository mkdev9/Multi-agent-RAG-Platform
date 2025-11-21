"""Database engine and session management."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import settings
from ..models.base import Base

logger = logging.getLogger(__name__)

# Global engine instance
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the database engine."""
    global _engine
    
    if _engine is None:
        _engine = create_async_engine(
            settings.database.url,
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # 1 hour
        )
        logger.info("Database engine created")
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Session factory created")
    
    return _session_factory


async def get_session() -> AsyncSession:
    """Get a database session."""
    factory = get_session_factory()
    return factory()


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Database dependency for FastAPI."""
    session = await get_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """Initialize the database tables."""
    engine = get_engine()
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created")


async def close_db() -> None:
    """Close the database connections."""
    global _engine, _session_factory
    
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")