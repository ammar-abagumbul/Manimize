from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

from pydantic import BaseModel, field_validator
from typing import Optional
from uuid import UUID as UUIDType

Base = declarative_base()

class Code(Base):
    __tablename__ = "codes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    code_content = Column(String, nullable=False)
    iteration = Column(Integer, nullable=False)
    is_faulty = Column(Boolean, default=False, nullable=False)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # Relationship to session
    session = relationship("Session", back_populates="codes")

    def __repr__(self):
        return (
            f"<Code(id={self.id}, session_id={self.session_id}, iteration={self.iteration}, "
            f"is_faulty={self.is_faulty}, created_at={self.created_at})>"
        )

class CodeBase(BaseModel):
    code_content: str
    iteration: int
    is_faulty: bool = False
    error_message: Optional[str] = None

    @field_validator("iteration")
    def iteration_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("Iteration must be greater than 0")
        return v

class CodeCreate(CodeBase):
    session_id: UUIDType

class CodeRead(CodeBase):
    id: UUIDType
    created_at: datetime
    session_id: UUIDType

    class Config:
        orm_mode = True
