from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

from pydantic import BaseModel, EmailStr, field_validator
from typing import List, Optional
from uuid import UUID as UUIDType

Base = declarative_base()

# SQLAlchemy User model
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationship to sessions
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"

# Pydantic models

class UserBase(BaseModel):
    username: str
    email: EmailStr

    @field_validator("username")
    def username_length(cls, v):
        if not (1 <= len(v) <= 50):
            raise ValueError("Username must be between 1 and 50 characters")
        return v

class UserCreate(UserBase):
    # Add password here if needed for registration, but do not store in DB directly
    pass

class UserRead(UserBase):
    id: UUIDType
    created_at: datetime
    updated_at: datetime
    sessions: Optional[List["SessionRead"]] = None  # Forward reference

    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None

    @field_validator("username")
    def username_length(cls, v):
        if v is not None and not (1 <= len(v) <= 50):
            raise ValueError("Username must be between 1 and 50 characters")
        return v

    class Config:
        orm_mode = True

# For forward references (SessionRead inside UserRead)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .session import SessionRead

UserRead.model_rebuild()
