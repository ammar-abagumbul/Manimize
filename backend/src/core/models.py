# models.py
# Data models for the Manimizer application

from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage

class Concept(BaseModel):
    """Model for a mathematical concept to be explained."""
    topic: str = Field(..., description="The name or topic of the concept")
    description: str = Field(..., description="Detailed explanation of the concept")

class ManimCode(BaseModel):
    """Model for generated Manim code."""
    code: str = Field(..., description="Manim code to visualize the concept")
    filename: Optional[str] = None  # Path to saved file, if written

class ChatMessage(BaseModel):
    """Model for chat messages between user and AI."""
    content: str = Field(..., description="Content of the message")
    is_user: bool = Field(..., description="True if message is from user, False if from AI")

class ConceptState(BaseModel):
    """State model for concept refinement process."""
    messages: List[ChatMessage] = []
    concept: Optional[Concept] = None

class ExecutionResult(BaseModel):
    """Model for results of Manim code execution."""
    success: bool = Field(..., description="Whether execution was successful")
    video_path: Optional[str] = None  # Path to generated video if successful
    error_message: Optional[str] = None  # Error details if execution failed
