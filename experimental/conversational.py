# DEPRECATED: This file has been modularized into smaller components.
# Please use the following files instead:
# - config.py: Configuration variables and constants
# - models.py: Data models and state definitions
# - utils.py: Utility functions for file operations and Manim execution
# - concept_refinement.py: Functions for refining mathematical concepts
# - manim_generation.py: Functions for generating Manim code
# - main.py: Main application entry point that ties everything together

import os
import subprocess

from pathlib import Path
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict

from pydantic import BaseModel, Field

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain.base_language import BaseLanguageModel

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph
from langgraph.graph import MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

AZURE_API = os.getenv("AZURE_API")
OUTPUT_FILE = Path.cwd() / "generated_manim.py"
LAST_ERROR_STATE = False
LOG_FILE = "./stdout.log"
MANIM_FILE = "./generated_manim.py"

manim_file = Path(MANIM_FILE)

openai = "openai/gpt-4.1"

assert AZURE_API is not None, "AZURE_API environment variable must be set"
print(f"Using Azure API: {AZURE_API}")

model = AzureAIChatCompletionsModel(
    model=openai,
    endpoint="https://models.github.ai/inference",
    credential=AZURE_API,
    temperature=0.0,
)


class Concept(BaseModel):
    """Concept to be explained by the model."""

    topic: str = Field(..., description="Name/topic of the concept to be explained")
    description: str = Field(..., description="Explanation of the concept")


class ManimCode(BaseModel):
    """Manim code to be generated for the concept."""

    code: str = Field(..., description="Manim code to visualize the concept")


class ConceptRefinementState(MessagesState):
    """State to hold the concept and its refinement."""

    concept: Concept


class ManimCodeState(TypedDict):
    """State to hold the Manim code and its execution."""

    manim_code: ManimCode
    error: bool


def write_to_file(
    code: Annotated[str, "manim code to be written"],
) -> str:
    """Useful to write a valid and executable manim code into a designated python file."""
    try:
        manim_file.write_text(code)
        return "successfully written to file"
    except Exception as e:
        return f"Error writing to file: {e}"


def run_manim_code(
    class_name: Annotated[str, "name of Manim scene for the code generated"]
) -> str:
    """Useful to execute the manim python code"""
    if not manim_file.exists():
        return "Path provided does not exist"
    command = [
        "manim",
        str(manim_file),
        class_name,
        "-ql",
        "--disable_caching",  # To ensure fresh render each time
    ]
    with open(LOG_FILE, "w") as log_file:
        try:
            subprocess.run(
                command,
                cwd=manim_file.parent,
                stdout=log_file,
                stderr=log_file,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return "Manim code did not run successfully!"
    return "Manim code executed successfully!"


def get_execution_output() -> str:
    """Get latest manim execution stdout and stderr. Only to be used in case latest execution fails with stderr"""
    log_file = Path("./stdout.log")
    return log_file.read_text()


concept_refinement_instructions = """You are an expert in explaining mathematical concepts clearly and concisely. 
1. First, review the topic {topic} and the description {description}. 
2. Examine the context and the desire voiced by the user to refine the explanation. The context is an interaction between the user and an AI assistant's attempt to generate a description that captures the topic and the user's description. 
{context}
3. Refine the description and the topic to make it clearer and upto the user's desire.
4. Your output should be a JSON object with the following schema: {{'concept': {{'topic': 'string', 'description': 'string'}}}} where topic is the name of the concept and description is the refined explanation of the concept.
"""


def refine_concept(state: ConceptRefinementState):
    response = None
    if len(state["messages"]) == 1:
        # Initial message, set up the questionn
        sys_message = SystemMessage(
            content="""You are an expert in explaining mathematical concepts. If user's request/question does not require some sort of mathematical descussion, you should respond with a very brief description of whichever concept you think is relevant. You shall always respond with json structured output with the following schema: {'concept': {'topic': 'string', 'description': 'string'}}.
Example Senario 1, user asks with little relevance for mathematical descussion:
User: Draw a triangle on the screen.
Assitant: {"concept": {"topic": "Triangle", "description": "A triangle is a polygon with three edges and three vertices. It is one of the basic shapes in geometry.""}}

Example Senario 2, user asks a question that requires a mathematical descussion:
User: What is the Pythagorean theorem?
Assitant: {"concept": {"topic": "Pythagorean theorem", "description": "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides."}}"""
        )
        response = model.with_structured_output(Concept, method="json_mode").invoke(
            [sys_message, HumanMessage(content=state["messages"][0].content)]
        )
        # WARN: not sure if `response` type fits messages type
    else:
        # Subsequent messages, refine the concepe
        sys_message = SystemMessage(
            concept_refinement_instructions.format(
                topic=state["concept"].topic,
                description=state["concept"].description,
                context="\n".join([msg.content for msg in state["messages"]]),
            )
        )

        response = model.with_structured_output(Concept, method="json_mode").invoke(
            [sys_message] + state["messages"]
        )
    print(response)
    return {
        "messages": [AIMessage(content=response["concept"]["description"])],
        "concept": Concept(
            topic=response["concept"]["topic"],
            description=response["concept"]["description"],
        ),
    }


def human_feedback(state: ConceptRefinementState):
    """Human feedback on the concept refinement."""
    user_feedback = input("User: ")
    return {
        "messages": [HumanMessage(content=user_feedback)],
    }


manim_code_instructions = """You are an expert in generating Manim code for mathematical concepts. Given the concept and its description, generate a Manim code that visualizes the concept. Ensure that you incorporate all elements of the description in the code to the best of your ability."""


def generate_manim_code(state: ConceptRefinementState):
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


def should_proceed_to_manim(state: ConceptRefinementState):
    schema = {
        "should_proceed": "a boolean value indicating whether the conversation should proceed to the Manim code generation step."
    }
    sys_message = SystemMessage(
        content="""You are an expert in determining whether a given conversation between a user and an AI assistant should proceed to the Manim code generation step. Your output should be a JSON object with the following schema: {'should_proceed': boolean} where boolean is either true or false"""
    )
    response = model.with_structured_output(schema, method="json_mode").invoke(
        [sys_message] + state["messages"]
    )
    if response["should_proceed"]:
        print("----------------Proceeding to Manim code generation----------------")
        return "generate_manim_code"
    return "human_feedback"


builder = StateGraph(ConceptRefinementState)
builder.add_node("refine_concept", refine_concept)
builder.add_node("human_feedback", human_feedback)
builder.add_node("generate_manim_code", generate_manim_code)

builder.set_entry_point("refine_concept")
builder.add_conditional_edges(
    "refine_concept", should_proceed_to_manim, ["generate_manim_code", "human_feedback"]
)
builder.add_edge("generate_manim_code", END)

builder.add_edge("human_feedback", "refine_concept")

graph = builder.compile()
graph.invoke(
    {
        "messages": [HumanMessage(content="What are rational numbers?")],
    }
)
