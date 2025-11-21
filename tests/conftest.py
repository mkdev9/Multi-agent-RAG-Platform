"""Test configuration and fixtures."""
import asyncio
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.database import get_db
from src.models.base import Base
from src.models.user import User
from src.core.auth import auth_service

# Test database URL (use in-memory SQLite for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest_asyncio.fixture
async def async_engine():
    """Create async engine for tests."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def async_session(async_engine):
    """Create async session for tests."""
    async_session_factory = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_factory() as session:
        yield session


@pytest.fixture
def client(async_session):
    """Create test client with database dependency override."""
    
    async def override_get_db():
        yield async_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(async_session):
    """Create test user."""
    user = User(
        email="test@example.com",
        hashed_password=auth_service.hash_password("testpassword123"),
        full_name="Test User",
        role="user",
        is_active=True,
    )
    
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def admin_user(async_session):
    """Create admin user."""
    user = User(
        email="admin@example.com",
        hashed_password=auth_service.hash_password("adminpassword123"),
        full_name="Admin User",
        role="admin",
        is_active=True,
    )
    
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    
    return user


@pytest.fixture
def auth_headers(test_user):
    """Create authorization headers for test user."""
    token_data = {
        "sub": str(test_user.id),
        "email": test_user.email,
        "role": test_user.role,
    }
    access_token = auth_service.create_access_token(token_data)
    
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def admin_headers(admin_user):
    """Create authorization headers for admin user."""
    token_data = {
        "sub": str(admin_user.id),
        "email": admin_user.email,
        "role": admin_user.role,
    }
    access_token = auth_service.create_access_token(token_data)
    
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()