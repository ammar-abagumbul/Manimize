from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

from pydantic import BaseModel, field_validator
from typing import Optional
from uuid import UUID as UUIDType

Base = declarative_base()

# SQLAlchemy Video model
class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        UniqueConstraint("session_id", name="uq_video_session_id"),
        UniqueConstraint("minio_object_key", name="uq_video_minio_object_key"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True)
    minio_bucket = Column(String(255), nullable=False, default="manimize-videos")
    minio_object_key = Column(String(255), nullable=False, unique=True)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(50), nullable=False)
    duration = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # Relationship to session
    session = relationship("Session", back_populates="video", uselist=False)

    def __repr__(self):
        return (
            f"<Video(id={self.id}, session_id={self.session_id}, minio_bucket={self.minio_bucket}, "
            f"minio_object_key={self.minio_object_key}, mime_type={self.mime_type}, created_at={self.created_at})>"
        )

# Pydantic models

class VideoBase(BaseModel):
    minio_bucket: str = "manimize-videos"
    minio_object_key: str
    file_size: Optional[int] = None
    mime_type: str
    duration: Optional[float] = None

    @field_validator("minio_bucket")
    def bucket_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("minio_bucket must not be empty")
        return v

    @field_validator("minio_object_key")
    def object_key_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("minio_object_key must not be empty")
        return v

    @field_validator("mime_type")
    def mime_type_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("mime_type must not be empty")
        return v

class VideoCreate(VideoBase):
    session_id: UUIDType

class VideoRead(VideoBase):
    id: UUIDType
    created_at: datetime
    session_id: UUIDType

    class Config:
        orm_mode = True
