"""Authentication and authorization core functionality."""
import uuid
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..config import settings
from ..models.user import User
from ..schemas.auth import UserCreate, TokenResponse


class AuthService:
    """Authentication service."""
    
    def __init__(self):
        self.secret_key = settings.auth.secret_key
        self.algorithm = settings.auth.algorithm
        self.access_token_expire_minutes = settings.auth.access_token_expire_minutes
        self.refresh_token_expire_days = settings.auth.refresh_token_expire_days
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    
    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=self.refresh_token_expire_days
            )
        
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )
            return payload
        except JWTError:
            return None
    
    async def authenticate_user(
        self,
        db: AsyncSession,
        email: str,
        password: str
    ) -> Optional[User]:
        """Authenticate user with email and password."""
        # Get user by email
        stmt = select(User).where(User.email == email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        if not user.is_active:
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()
        
        return user
    
    async def create_user(
        self,
        db: AsyncSession,
        user_create: UserCreate
    ) -> User:
        """Create new user."""
        # Hash password
        hashed_password = self.hash_password(user_create.password)
        
        # Create user
        db_user = User(
            email=user_create.email,
            hashed_password=hashed_password,
            full_name=user_create.full_name,
            role=user_create.role,
            is_active=user_create.is_active,
        )
        
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        
        return db_user
    
    async def get_user_by_id(
        self,
        db: AsyncSession,
        user_id: uuid.UUID
    ) -> Optional[User]:
        """Get user by ID."""
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_user_by_email(
        self,
        db: AsyncSession,
        email: str
    ) -> Optional[User]:
        """Get user by email."""
        stmt = select(User).where(User.email == email)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create_token_response(
        self,
        user: User
    ) -> TokenResponse:
        """Create token response for user."""
        # Create token data
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
        }
        
        # Create tokens
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token({"sub": str(user.id)})
        
        from ..schemas.auth import UserResponse
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60,
            user=UserResponse.model_validate(user),
        )


# Global auth service instance
auth_service = AuthService()