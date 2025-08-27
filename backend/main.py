from fastapi import FastAPI, Depends
from pydantic import BaseModel
from pocketflow import Flow
from typing import Optional
from src.agent.nodes import (
    GenerateOutlineNode,
    ManimCodeGenerationNode,
    ErrorResolver,
    ManimExecutor,
)
from src.models.user import User
from dependencies import get_current_user

# --- set logging ---
import logging
from logging.config import dictConfig
from config.dict_config import log_config

dictConfig(log_config)

logger = logging.getLogger("fastapi")

class ChatRequest(BaseModel):
    user_query: str

app = FastAPI()

@app.post("/")
async def root(request: ChatRequest, user: Optional[User] = Depends(get_current_user)):
    user_query = request.user_query

    generate_outline = GenerateOutlineNode()
    generate_manim   = ManimCodeGenerationNode()
    resolve_error    = ErrorResolver()
    execute_manim    = ManimExecutor()

    generate_outline >> generate_manim >> execute_manim
    execute_manim - "correct_error" >> resolve_error
    resolve_error >> execute_manim

    flow = Flow(start=generate_outline)
    shared = {"user_query": user_query}
    flow.run(shared)

    return {"response": shared.get("outline", "No response was returned")}
