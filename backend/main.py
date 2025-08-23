from fastapi import FastAPI
from pydantic import BaseModel
from pocketflow import Flow
from src.agent.nodes import (
    GenerateOutlineNode,
    ManimCodeGenerationNode,
    ErrorResolver,
    ManimExecutor,
)

from src.dcelery.tasks import render_manim_scene

class ChatRequest(BaseModel):
    user_query: str

app = FastAPI()

@app.post("/render")
async def test():
    res = render_manim_scene.delay()
    return {"result": res}


@app.post("/")
async def root(request: ChatRequest):
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

    print("---------------------- Shared ----------------------")
    print(shared.keys())
    print(shared)
    print("---------------------- Shared ----------------------")

    return {"response": shared.get("outline", "No response was returned")}
