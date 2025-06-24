# concept_agent.py
# Logic for concept refinement agent

from langchain.base_language import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel

from ..core.models import Concept, ConceptState, ChatMessage

class ConceptResponse(BaseModel):
    concept: Concept

def refine_concept(state: ConceptState, model: BaseLanguageModel) -> dict:
    """Refine a mathematical concept based on the current chat state."""
    system_prompt = """You are an expert in explaining mathematical concepts clearly and concisely. 
Respond to the user's queries about math concepts with detailed yet accessible explanations. 
If the user provides external sources or context, incorporate them into your explanation. 
Your goal is to help the user understand the concept thoroughly before visualizing it with Manim.
Always output a JSON object with the schema: {"concept": {"topic": "string", "description": "string"}}."""

    messages = []
    for msg in state.messages:
        if msg.is_user:
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    if not state.messages:
        messages.append(HumanMessage(content="Hello, I'm ready to help you understand mathematical concepts. Please ask a question or describe a concept you'd like to explore."))

    response = model.with_structured_output(ConceptResponse, method="json_mode").invoke(
        [SystemMessage(content=system_prompt)] + messages
    )

    new_message = ChatMessage(content=response.concept.description, is_user=False)
    updated_messages = state.messages + [new_message]
    return {
        "messages": updated_messages,
        "concept": response.concept
    }

def should_proceed_to_manim(state: ConceptState, model: BaseLanguageModel) -> bool:
    """Determine if the conversation should proceed to Manim code generation."""
    system_prompt = """You are an expert in determining whether a conversation about a mathematical concept is ready for visualization with Manim. 
Review the chat history and decide if the user's understanding is sufficient to proceed. 
Output a JSON object with the schema: {"should_proceed": boolean}."""

    class ProceedResponse(BaseModel):
        should_proceed: bool

    messages = []
    for msg in state.messages:
        if msg.is_user:
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    response = model.with_structured_output(ProceedResponse, method="json_mode").invoke(
        [SystemMessage(content=system_prompt)] + messages
    )
    return response.should_proceed
