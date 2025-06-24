# graph.py
# Langgraph workflow definition for Manimizer application

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.models import ConceptState, ManimCode, ExecutionResult
from ..agents.concept_agent import refine_concept, should_proceed_to_manim
from ..agents.manim_agent import generate_manim_code
from ..agents.debug_agent import debug_manim_code
from ..execution.manim_executor import write_and_execute_manim_code
from ..core.config import MAX_DEBUG_ATTEMPTS

def build_workflow(model):
    """Build the Langgraph workflow for concept refinement, Manim generation, and execution."""
    builder = StateGraph(ConceptState)
    
    # Add nodes
    builder.add_node("refine_concept", lambda state: refine_concept(state, model))
    builder.add_node("generate_manim_code", lambda state: {"manim_code": generate_manim_code(state.concept, model)})
    builder.add_node("execute_manim_code", lambda state: {"execution_result": write_and_execute_manim_code(state.manim_code, model)})
    
    # Add conditional edges
    builder.add_conditional_edges(
        "refine_concept",
        lambda state: should_proceed_to_manim(state, model),
        {True: "generate_manim_code", False: "refine_concept"}
    )
    
    def should_debug(state):
        """Determine if debugging is needed based on execution result and retry count."""
        if not state.execution_result.success:
            debug_attempts = getattr(state, "debug_attempts", 0)
            return debug_attempts < MAX_DEBUG_ATTEMPTS
        return False
    
    builder.add_conditional_edges(
        "execute_manim_code",
        should_debug,
        {True: "debug_code", False: END}
    )
    
    builder.add_node("debug_code", lambda state: {
        "manim_code": debug_manim_code(state.manim_code, state.execution_result.error_message, model),
        "debug_attempts": getattr(state, "debug_attempts", 0) + 1
    })
    builder.add_edge("debug_code", "execute_manim_code")
    
    # Set entry point
    builder.set_entry_point("refine_concept")
    
    # Compile the graph with memory saver for persistence
    graph = builder.compile(checkpointer=MemorySaver())
    return graph
