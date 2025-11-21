"""Authentication schemas."""
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from .common import BaseSchema


class UserBase(BaseSchema):
    """Base user schema."""
    
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User full name")
    role: str = Field(default="user", description="User role")
    is_active: bool = Field(default=True, description="User active status")


class UserCreate(UserBase):
    """User creation schema."""
    
    password: str = Field(..., min_length=8, description="User password")


class UserUpdate(BaseSchema):
    """User update schema."""
    
    full_name: Optional[str] = Field(None, description="User full name")
    role: Optional[str] = Field(None, description="User role")
    is_active: Optional[bool] = Field(None, description="User active status")


class UserResponse(UserBase):
    """User response schema."""
    
    id: uuid.UUID = Field(..., description="User ID")
    is_verified: bool = Field(..., description="User verification status")
    last_login: Optional[datetime] = Field(None, description="Last login time")
    created_at: datetime = Field(..., description="Account creation time")
    updated_at: datetime = Field(..., description="Account last update time")


class LoginRequest(BaseSchema):
    """Login request schema."""
    
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")


class TokenResponse(BaseSchema):
    """Token response schema."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: UserResponse = Field(..., description="User information")


class RefreshTokenRequest(BaseSchema):
    """Refresh token request schema."""
    
    refresh_token: str = Field(..., description="Refresh token")


class PasswordChangeRequest(BaseSchema):
    """Password change request schema."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class PasswordResetRequest(BaseSchema):
    """Password reset request schema."""
    
    email: EmailStr = Field(..., description="User email")


class PasswordResetConfirm(BaseSchema):
    """Password reset confirmation schema."""
    
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")


class APIKeyCreate(BaseSchema):
    """API key creation schema."""
    
    name: str = Field(..., description="API key name")
    permissions: Optional[list[str]] = Field(
        default_factory=list,
        description="API key permissions"
    )
    expires_at: Optional[datetime] = Field(None, description="Expiration date")


class APIKeyResponse(BaseSchema):
    """API key response schema."""
    
    id: uuid.UUID = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    key: str = Field(..., description="API key (only shown once)")
    permissions: list[str] = Field(..., description="API key permissions")
    is_active: bool = Field(..., description="API key status")
    created_at: datetime = Field(..., description="Creation time")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    last_used: Optional[datetime] = Field(None, description="Last used time")