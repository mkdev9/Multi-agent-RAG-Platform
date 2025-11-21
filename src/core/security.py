"""Security utilities and middleware."""
import uuid
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.user import User
from .auth import auth_service

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify token
    payload = auth_service.verify_token(credentials.credentials)
    if payload is None:
        raise credentials_exception
    
    # Check token type
    if payload.get("type") != "access":
        raise credentials_exception
    
    # Get user ID
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception
    
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise credentials_exception
    
    # Get user from database
    user = await auth_service.get_user_by_id(db, user_id)
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_role(required_role: str):
    """Dependency to require specific role."""
    async def check_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return check_role


def require_admin():
    """Dependency to require admin role."""
    return require_role("admin")


class RoleChecker:
    """Role checker class for more complex role requirements."""
    
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {self.allowed_roles}"
            )
        return current_user


# Common role checkers
admin_required = RoleChecker(["admin"])
power_user_required = RoleChecker(["admin", "power_user"])
user_or_above = RoleChecker(["admin", "power_user", "user"])


async def verify_refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Verify refresh token and return user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify token
    payload = auth_service.verify_token(credentials.credentials)
    if payload is None:
        raise credentials_exception
    
    # Check token type
    if payload.get("type") != "refresh":
        raise credentials_exception
    
    # Get user ID
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception
    
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise credentials_exception
    
    # Get user from database
    user = await auth_service.get_user_by_id(db, user_id)
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


def get_optional_user():
    """Get current user if authenticated, otherwise None."""
    async def _get_optional_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        db: AsyncSession = Depends(get_db)
    ) -> Optional[User]:
        if credentials is None:
            return None
        
        try:
            return await get_current_user(credentials, db)
        except HTTPException:
            return None
    
    return _get_optional_user