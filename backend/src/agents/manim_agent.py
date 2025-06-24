# manim_agent.py
# Logic for Manim code generation agent

from langchain.base_language import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..core.models import Concept, ManimCode

class ManimCodeResponse(BaseModel):
    manim_code: ManimCode

def generate_manim_code(concept: Concept, model: BaseLanguageModel) -> ManimCode:
    """Generate Manim code to visualize the given mathematical concept."""
    system_prompt = """You are an expert in generating Manim code for visualizing mathematical concepts. 
Given a concept and its detailed description, create Manim code that accurately represents the concept. 
Ensure the code is complete, includes necessary imports, and follows Manim best practices for clarity and animation. 
Always output a JSON object with the schema: {"manim_code": {"code": "string"}}."""

    user_message = f"Generate Manim code for the concept '{concept.topic}' with the following description: {concept.description}"
    
    response = model.with_structured_output(ManimCodeResponse, method="json_mode").invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
    )
    
    return response.manim_code
