from fastapi import FastAPI
from pydantic import BaseModel
from pocketflow import Flow
from src.agent.workflow_pocketflow import (
    GenerateOutlineNode,
    ManimCodeGenerationNode,
    ErrorResolver,
    ManimExecutor,
    END
)

class ChatRequest(BaseModel):
    user_query: str

app = FastAPI()

@app.post("/")
async def root(request: ChatRequest):
    user_query = request.user_query

    generate_outline = GenerateOutlineNode()
    generate_manim   = ManimCodeGenerationNode()
    resolve_error    = ErrorResolver()
    execute_manim    = ManimExecutor()
    end              = END()

    generate_outline >> generate_manim >> execute_manim
    execute_manim - "correct_error" >> resolve_error
    execute_manim - "default" >> end
    resolve_error >> execute_manim

    flow = Flow(start=generate_outline)
    shared = {"user_query": user_query}
    flow.run(shared)

    print("---------------------- Shared ----------------------")
    print(shared.keys())
    print(shared)
    print("---------------------- Shared ----------------------")

    return {"response": shared.get("outline", "No response was returned")}
