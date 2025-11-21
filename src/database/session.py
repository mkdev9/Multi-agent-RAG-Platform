"""Database session dependency."""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .engine import get_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Database dependency for FastAPI."""
    session_factory = get_session_factory()
    session = session_factory()
    
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()