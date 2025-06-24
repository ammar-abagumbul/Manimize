# debug_agent.py
# Logic for debugging Manim code

from langchain.base_language import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..core.models import ManimCode

class DebugResponse(BaseModel):
    manim_code: ManimCode

def debug_manim_code(manim_code: ManimCode, error_message: str, model: BaseLanguageModel) -> ManimCode:
    """Debug and revise Manim code based on the error message."""
    system_prompt = """You are an expert in debugging Manim code. 
Given a piece of Manim code and an error message from a failed execution, analyze the error and revise the code to fix the issue. 
Ensure the revised code is complete and adheres to Manim best practices. 
Always output a JSON object with the schema: {"manim_code": {"code": "string"}}."""

    user_message = f"The following Manim code failed to execute with the error message below:\n\nCode:\n{manim_code.code}\n\nError Message:\n{error_message}\n\nPlease revise the code to fix the error."
    
    response = model.with_structured_output(DebugResponse, method="json_mode").invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
    )
    
    return response.manim_code
