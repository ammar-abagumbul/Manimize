# FastAPI application entry point for Manimizer

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.config import API_KEY, MODEL_ENDPOINT, MODEL_NAME, TEMPERATURE
from src.api.routes import router as api_router
from src.workflow.graph import build_workflow
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

app = FastAPI(title="Manimizer API", description="API for generating Manim visualizations of mathematical concepts")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])

# Initialize model and workflow on startup using lifespan events
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the language model and workflow graph on startup."""
    from src.api.routes import model, workflow_graph
    
    # Initialize the Azure AI model (or other model as configured)
    model_instance = AzureAIChatCompletionsModel(
        model=MODEL_NAME,
        endpoint=MODEL_ENDPOINT,
        credential=API_KEY,
        temperature=TEMPERATURE,
    )
    
    # Build the workflow graph
    graph = build_workflow(model_instance)
    
    # Assign to router for use in endpoints
    model = model_instance
    workflow_graph = graph
    print("Model and workflow initialized successfully.")
    try:
        yield
    finally:
        # Any shutdown logic can go here if needed
        pass

app = FastAPI(title="Manimizer API", description="API for generating Manim visualizations of mathematical concepts", lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
