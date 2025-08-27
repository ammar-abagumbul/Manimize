from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

from typing import Optional, List, TYPE_CHECKING
from uuid import UUID as UUIDType

from pydantic import BaseModel, field_validator
from .enums import SessionStatus

Base = declarative_base()

# SQLAlchemy Session model
class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True) # Allow anonymous sessions
    query = Column(String(500), nullable=False)
    ai_response = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default=SessionStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="SET NULL"), nullable=True, unique=True)

    # Relationships
    user = relationship("User", back_populates="sessions")
    codes = relationship("Code", back_populates="session", cascade="all, delete-orphan")
    video = relationship("Video", back_populates="session", uselist=False)

    def __repr__(self):
        return (
            f"<Session(id={self.id}, user_id={self.user_id}, status={self.status}, "
            f"created_at={self.created_at}, updated_at={self.updated_at}, video_id={self.video_id})>"
        )

# Pydantic models

if TYPE_CHECKING:
    from ..models.user import UserRead
    from ..models.code import CodeRead
    from ..models.video import VideoRead

class SessionBase(BaseModel):
    query: str
    ai_response: str
    status: SessionStatus = SessionStatus.PENDING

    @field_validator("query")
    def query_length(cls, v):
        if not v or len(v) > 500:
            raise ValueError("Query must be non-empty and at most 500 characters")
        return v

class SessionCreate(SessionBase):
    user_id: UUIDType

class SessionUpdate(BaseModel):
    query: Optional[str] = None
    ai_response: Optional[str] = None
    status: Optional[SessionStatus] = None
    video_id: Optional[UUIDType] = None

    @field_validator("query")
    def query_length(cls, v):
        if v is not None and len(v) > 500:
            raise ValueError("Query must be at most 500 characters")
        return v

    class Config:
        orm_mode = True

class SessionRead(SessionBase):
    id: UUIDType
    created_at: datetime
    updated_at: datetime
    user: Optional["UserRead"]
    codes: Optional[List["CodeRead"]] = None
    video: Optional["VideoRead"] = None

    class Config:
        orm_mode = True

SessionRead.model_rebuild()
