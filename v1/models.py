from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import MessagesState

class Concept(BaseModel):
    """Concept to be explained by the model."""

    topic: str = Field(..., description="Name/topic of the concept to be explained")
    description: str = Field(..., description="Explanation of the concept")


class ManimCode(BaseModel):
    """Manim code to be generated for the concept."""

    code: str = Field(..., description="Manim code to visualize the concept")


class ConceptRefinementState(MessagesState):
    """State to hold the concept and its refinement."""

    concept: Concept


class ManimCodeState(TypedDict):
    """State to hold the Manim code and its execution."""

    manim_code: ManimCode
    error: bool
