"""Authentication routes."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...schemas.auth import (
    UserCreate, UserResponse, LoginRequest, TokenResponse, 
    RefreshTokenRequest, PasswordChangeRequest
)
from ...schemas.common import SuccessResponse, ErrorResponse
from ...core.auth import auth_service
from ...core.security import get_current_user, verify_refresh_token
from ...core.exceptions import AuthenticationError, ConflictError
from ...models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse)
async def register_user(
    user_create: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user."""
    # Check if user already exists
    existing_user = await auth_service.get_user_by_email(db, user_create.email)
    if existing_user:
        raise ConflictError("User with this email already exists")
    
    # Create user
    user = await auth_service.create_user(db, user_create)
    
    # Create token response
    token_response = await auth_service.create_token_response(user)
    
    return token_response


@router.post("/login", response_model=TokenResponse)
async def login_user(
    login_request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """Login user and return tokens."""
    # Authenticate user
    user = await auth_service.authenticate_user(
        db, login_request.email, login_request.password
    )
    
    if not user:
        raise AuthenticationError("Invalid email or password")
    
    # Create token response
    token_response = await auth_service.create_token_response(user)
    
    return token_response


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token."""
    # Verify refresh token
    payload = auth_service.verify_token(refresh_request.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise AuthenticationError("Invalid refresh token")
    
    # Get user
    user_id = payload.get("sub")
    user = await auth_service.get_user_by_id(db, user_id)
    
    if not user:
        raise AuthenticationError("User not found")
    
    # Create new token response
    token_response = await auth_service.create_token_response(user)
    
    return token_response


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information."""
    return UserResponse.model_validate(current_user)


@router.post("/change-password", response_model=SuccessResponse)
async def change_password(
    password_change: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change user password."""
    # Verify current password
    if not auth_service.verify_password(
        password_change.current_password, 
        current_user.hashed_password
    ):
        raise AuthenticationError("Current password is incorrect")
    
    # Update password
    new_hashed_password = auth_service.hash_password(password_change.new_password)
    current_user.hashed_password = new_hashed_password
    
    await db.commit()
    
    return SuccessResponse(message="Password changed successfully")


@router.post("/logout", response_model=SuccessResponse)
async def logout_user(
    current_user: User = Depends(get_current_user)
):
    """Logout user (client-side token invalidation)."""
    # In a production system, you might want to maintain a blacklist of tokens
    # For now, we'll just return success and let the client handle token removal
    return SuccessResponse(message="Logged out successfully")