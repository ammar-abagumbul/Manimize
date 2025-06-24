from langchain_core.messages import HumanMessage, SystemMessage
from langchain.base_language import BaseLanguageModel

from models import ConceptRefinementState

# Instructions for Manim code generation
manim_code_instructions = """You are an expert in generating Manim code for mathematical concepts. Given the concept and its description, generate a Manim code that visualizes the concept. Ensure that you incorporate all elements of the description in the code to the best of your ability."""

def generate_manim_code(state: ConceptRefinementState, model: BaseLanguageModel):
    """Generate Manim code for the concept."""
    topic = state["concept"].topic
    description = state["concept"].description
    msg = f"""Given the concept {topic} and its description {description}, generate a Manim code that visualizes the concept."""
    result = model.invoke(
        [SystemMessage(content=manim_code_instructions), HumanMessage(content=msg)]
    )
    print(result.content)
    return {
        "messages": [result],
        "manim_code": result.content,
    }
