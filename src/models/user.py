"""User model."""
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255))
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(default=False, nullable=False)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    def __repr__(self) -> str:
        return f"<User(email={self.email}, role={self.role})>"