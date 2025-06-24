# routes.py
# FastAPI route definitions for Manimizer application

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel
from typing import List, Optional

from ..core.models import ConceptState, ChatMessage, ExecutionResult
from ..workflow.graph import build_workflow

router = APIRouter()

# Placeholder for the model and workflow graph
# In a real implementation, these would be initialized properly with dependency injection
model = None  # To be set during app startup
workflow_graph = None  # To be set during app startup

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # For maintaining conversation state

class ChatResponse(BaseModel):
    message: str
    session_id: str
    messages: List[ChatMessage]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle user chat input for concept refinement."""
    if not model or not workflow_graph:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Retrieve or initialize session state
    session_id = request.session_id or str(uuid.uuid4())
    state = workflow_graph.get_state(session_id) if request.session_id else ConceptState()
    
    # Add user message to state
    new_message = ChatMessage(content=request.message, is_user=True)
    state.messages.append(new_message)
    
    # Run the workflow
    updated_state = await workflow_graph.arun(state, session_id=session_id)
    
    # Return the latest AI response
    latest_message = updated_state.messages[-1].content if updated_state.messages else "No response generated."
    return ChatResponse(
        message=latest_message,
        session_id=session_id,
        messages=updated_state.messages
    )

class GenerateManimRequest(BaseModel):
    session_id: str

class GenerateManimResponse(BaseModel):
    status: str
    session_id: str
    manim_code: Optional[str] = None

@router.post("/generate-manim", response_model=GenerateManimResponse)
async def generate_manim(request: GenerateManimRequest):
    """Trigger Manim code generation for the refined concept."""
    if not model or not workflow_graph:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    state = workflow_graph.get_state(request.session_id)
    if not state or not state.concept:
        raise HTTPException(status_code=400, detail="No refined concept found for this session")
    
    # Check if should proceed to Manim generation
    from ..agents.concept_agent import should_proceed_to_manim
    if not should_proceed_to_manim(state, model):
        raise HTTPException(status_code=400, detail="Concept refinement not complete")
    
    # Run workflow to generate Manim code
    updated_state = await workflow_graph.arun(state, session_id=request.session_id)
    manim_code = updated_state.manim_code.code if hasattr(updated_state, "manim_code") else None
    
    return GenerateManimResponse(
        status="Manim code generated" if manim_code else "Failed to generate Manim code",
        session_id=request.session_id,
        manim_code=manim_code
    )

class StatusRequest(BaseModel):
    session_id: str

class StatusResponse(BaseModel):
    status: str
    session_id: str
    video_url: Optional[str] = None
    error_message: Optional[str] = None

@router.get("/status", response_model=StatusResponse)
async def get_status(request: StatusRequest):
    """Get the status of Manim code execution."""
    if not workflow_graph:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    state = workflow_graph.get_state(request.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if hasattr(state, "execution_result") and state.execution_result.success:
        return StatusResponse(
            status="Video generated",
            session_id=request.session_id,
            video_url=state.execution_result.video_path
        )
    elif hasattr(state, "execution_result") and not state.execution_result.success:
        return StatusResponse(
            status="Execution failed",
            session_id=request.session_id,
            error_message=state.execution_result.error_message
        )
    elif hasattr(state, "manim_code"):
        return StatusResponse(
            status="Executing Manim code",
            session_id=request.session_id
        )
    else:
        return StatusResponse(
            status="Concept refinement in progress",
            session_id=request.session_id
        )

@router.get("/videos", response_model=List[str])
async def list_videos():
    """List available generated videos."""
    from ..core.config import OUTPUT_VIDEOS_DIR
    from ..core.utils import ensure_directory
    import os
    
    ensure_directory(OUTPUT_VIDEOS_DIR)
    videos = [f"/videos/{f}" for f in os.listdir(OUTPUT_VIDEOS_DIR) if f.endswith('.mp4')]
    return videos
