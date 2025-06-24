from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config import AZURE_API, OPENAI_MODEL, OPENAI_ENDPOINT, TEMPERATURE
from models import ConceptRefinementState
from concept_refinement import refine_concept, human_feedback, should_proceed_to_manim
from manim_generation import generate_manim_code

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel


def main():
    """Main function to run the manimizer application."""
    # Initialize the model
    model = AzureAIChatCompletionsModel(
        model=OPENAI_MODEL,
        endpoint=OPENAI_ENDPOINT,
        credential=AZURE_API,
        temperature=TEMPERATURE,
    )

    # Build the graph
    builder = StateGraph(ConceptRefinementState)
    
    # Add nodes with model dependency
    builder.add_node("refine_concept", lambda state: refine_concept(state, model))
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("generate_manim_code", lambda state: generate_manim_code(state, model))

    # Set entry point and edges
    builder.set_entry_point("refine_concept")
    builder.add_conditional_edges(
        "refine_concept", 
        lambda state: should_proceed_to_manim(state, model), 
        ["generate_manim_code", "human_feedback"]
    )
    builder.add_edge("generate_manim_code", END)
    builder.add_edge("human_feedback", "refine_concept")

    # Compile the graph
    graph = builder.compile()
    
    # Run the graph with an initial message
    initial_message = input("Enter your mathematical concept question: ")
    graph.invoke(
        {
            "messages": [HumanMessage(content=initial_message)],
        }
    )


if __name__ == "__main__":
    main()
