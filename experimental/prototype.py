import os
import json
import uuid
import subprocess
from dotenv import load_dotenv
from typing import Optional, Dict
from langchain_core.messages.ai import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

load_dotenv()
AZURE_API = os.getenv("AZURE_API")


class AppState(BaseModel):
    user_query: str = Field(description="User's math concept query")
    outline: Optional[Dict] = Field(default=None, description="High-level outline")
    explanation: Optional[str] = Field(
        default=None, description="Detailed narrative script"
    )
    manim_code: Optional[str] = Field(default=None, description="Generated Manim code")
    video_path: Optional[str] = Field(
        default=None, description="Path to rendered video"
    )
    error: Optional[str] = Field(default=None, description="Any errors encountered")
    feedback: Optional[str] = Field(default=None, description="Feedback from judge")
    code_file_path: Optional[str] = Field(
        default=None, description="Path to written Manim code file"
    )
    adherence_iterations: int = Field(
        default=0, description="Iterations for adherence refinement"
    )
    error_iterations: int = Field(
        default=0, description="Iterations for error refinement"
    )


class InteractionHandler(ABC):
    @abstractmethod
    def get_user_input(self) -> str:
        pass

    @abstractmethod
    def display_output(self, output: str) -> None:
        pass

    @abstractmethod
    def display_error(self, error: str) -> None:
        pass


class TerminalHandler(InteractionHandler):
    def get_user_input(self) -> str:
        return input("User ('quit' to exit): ")

    def display_output(self, output: str) -> None:
        print(output)

    def display_error(self, error: str) -> None:
        print(f"Error: {error}")


class LLMProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name == "grok":
            self.llm = AzureAIChatCompletionsModel(
                model="xai/grok-3-mini",
                endpoint="https://models.github.ai/inference",
                credential=AZURE_API,
                temperature=0.0,
            )
        elif model_name == "gpt-4o":
            self.llm = AzureAIChatCompletionsModel(
                model="openai/gpt-4.1",
                endpoint="https://models.github.ai/inference",
                credential=AZURE_API,
                temperature=0.0,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def invoke(self, prompt):
        return self.llm.invoke(prompt)


def query_input_node(state: AppState, handler: InteractionHandler) -> AppState | None:
    query = handler.get_user_input()
    if query.lower() == "quit":
        return None  # Signal to exit
    state.user_query = query
    return state


def outline_generation_node(state: AppState, llm_provider: LLMProvider) -> AppState:
    print("Generating outline...")
    outline_prompt = ChatPromptTemplate(
        [
            SystemMessage(
                content="Generate a JSON outline for explaining {concept} in 3Blue1Brown style. Structure: {'sections': [{'title': 'str', 'description': 'str', 'visual_ideas': 'str'}]}. Use visual ideas."
            ),
            HumanMessage(content=state.user_query),
        ]
    )

    response = llm_provider.invoke(
        outline_prompt.invoke({"concept": state.user_query})
    )

    outline_data = json.loads(str(response.content))
    state.outline = outline_data
    return state


def explanation_generation_node(state: AppState, llm_provider: LLMProvider) -> AppState:
    print("Generating explanation...")
    explanation_prompt = ChatPromptTemplate(
        [
            SystemMessage(
                content='Based on this outline: {outline}, generate a detailed narrative explanation for {concept} in 3Blue1Brown style. Output JSON: {"explanation": "full text here"}.'
            ),
            HumanMessage(content=state.user_query),
        ]
    )

    response = llm_provider.invoke(
        explanation_prompt.invoke(
            {"concept": state.user_query, "outline": json.dumps(state.outline)}
        )
    )

    exp_data = json.loads(str(response.content))
    state.explanation = exp_data["explanation"]
    return state


def manim_code_generation_node(state: AppState, llm_provider: LLMProvider) -> AppState:
    print("Generating Manim code...")
    code_prompt = ChatPromptTemplate(
        [
            SystemMessage(
                content="Generate Manim code to visualize this explanation: {explanation} for {concept}. Use class ExplanationScene(Scene): and import from manim import *. Output JSON: {'code': 'full python code here'}. Make sure you follow the proper JSON format where you use double quotes for keys and values. Do not surround the code with triple backticks or quotes."
            ),
            HumanMessage(content=state.user_query),
        ]
    )
    response = llm_provider.invoke(
        code_prompt.invoke(
            {"concept": state.user_query, "explanation": state.explanation}
        )
    )
    code_data = json.loads(str(response.content))
    state.manim_code = code_data["code"]
    return state


def judge_adherence_node(state: AppState, llm_provider: LLMProvider) -> AppState:
    print("Checking adherence to outline and explanation...")
    if state.adherence_iterations > 3:
        state.error = "Max adherence iterations exceeded."
        return state

    if state.manim_code is None:
        raise Exception("Checking adherence without a generated manim code.")

    judge_prompt = ChatPromptTemplate(
        [
            SystemMessage(
                content="Does this Manim code fully adhere to and cover all sections in the outline: {outline} and explanation: {explanation}? Output JSON: {'adheres': true/false, 'feedback': 'reasons if false'}."
            ),
            HumanMessage(content=state.manim_code),
        ]
    )
    # formatting is not working properly here

    response = llm_provider.invoke(
        judge_prompt.invoke(
            {
                "outline": json.dumps(state.outline),
                "explanation": state.explanation,
            }
        )
    )

    judge_data = json.loads(str(response.content))
    if judge_data["adheres"]:
        state.feedback = None
    else:
        state.feedback = judge_data["feedback"]
        state.adherence_iterations += 1
    return state


def adherence_refinement_node(state: AppState, llm_provider: LLMProvider) -> AppState:
    if state.manim_code is None:
        raise Exception("Checking adherence without a generated manim code.")

    print("Refining Manim code based on feedback...")

    refine_prompt = ChatPromptTemplate(
        [
            SystemMessage(
                content="Improve this Manim code to better adhere to the outline and explanation based on feedback: {feedback}. Output JSON: {'code': 'updated python code'}."
            ),
            HumanMessage(content=state.manim_code),
        ]
    )
    response = llm_provider.invoke(
        refine_prompt.invoke({"feedback": state.feedback})
    )

    code_data = json.loads(str(response.content))
    state.manim_code = code_data["code"]
    state.feedback = None
    return state


def write_code_to_file_node(state: AppState) -> AppState:
    if state.manim_code is None:
        raise Exception("Checking adherence without a generated manim code.")

    print("Writing Manim code to file...")
    scripts_dir = "manim_scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    file_name = f"{uuid.uuid4()}.py"
    state.code_file_path = os.path.join(scripts_dir, file_name)
    with open(state.code_file_path, "w") as f:
        f.write(state.manim_code)
    return state


def execute_manim_node(state: AppState) -> AppState:
    print("Executing Manim code...")
    if state.error_iterations > 3:
        state.error = "Max error iterations exceeded."
        return state

    if state.code_file_path is None:
        raise Exception("Manim file path is not set.")

    try:
        result = subprocess.run(
            ["manim", "-ql", state.code_file_path, "ExplanationScene"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            state.error = result.stderr
            state.error_iterations += 1
            return state
        # Parse video path from stdout
        output_lines = result.stdout.splitlines()
        video_path = None
        for line in output_lines:
            if line.startswith("File ready at "):
                video_path = line.split("File ready at ", 1)[1].strip()
                break
        if video_path:
            state.video_path = video_path
            state.error = None
        else:
            state.error = "Video path not found in Manim output."
    except Exception as e:
        state.error = str(e)
    return state


def error_refinement_node(state: AppState, llm_provider: LLMProvider) -> AppState:
    print("Refining Manim code based on execution error...")

    if state.manim_code is None:
        raise Exception("Checking adherence without a generated manim code.")

    refine_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Fix this Manim code based on the execution error: {error}. Output JSON: {'code': 'updated python code'}."
            ),
            HumanMessage(content=state.manim_code),
        ]
    )
    response = llm_provider.invoke(
        refine_prompt.invoke({"error": state.error})
    )
    code_data = json.loads(str(response.content))
    state.manim_code = code_data["code"]
    state.error = None
    return state


# --- Routers ---
def adherence_router(state: AppState) -> str:
    print("Adherence router...")
    if state.feedback is None and state.error is None:
        return "adheres"
    else:
        return "refine"


def execute_router(state: AppState) -> str:
    print("Execute router...")
    if state.error is None and state.video_path:
        return "success"
    else:
        return "refine"


def build_graph(handler: InteractionHandler, llm_provider: LLMProvider) -> CompiledStateGraph:
    graph = StateGraph(AppState)

    graph.add_node("query_input", lambda state: query_input_node(state, handler))
    graph.add_node(
        "generate_outline", lambda state: outline_generation_node(state, llm_provider)
    )
    graph.add_node(
        "generate_explanation",
        lambda state: explanation_generation_node(state, llm_provider),
    )
    graph.add_node(
        "generate_manim_code",
        lambda state: manim_code_generation_node(state, llm_provider),
    )
    graph.add_node(
        "judge_adherence", lambda state: judge_adherence_node(state, llm_provider)
    )
    graph.add_node(
        "adherence_refinement",
        lambda state: adherence_refinement_node(state, llm_provider),
    )
    graph.add_node("write_code_to_file", write_code_to_file_node)
    graph.add_node("execute_manim", execute_manim_node)
    graph.add_node(
        "error_refinement", lambda state: error_refinement_node(state, llm_provider)
    )

    graph.set_entry_point("query_input")
    graph.add_edge("query_input", "generate_outline")
    graph.add_edge("generate_outline", "generate_explanation")
    graph.add_edge("generate_explanation", "generate_manim_code")
    graph.add_edge("generate_manim_code", "judge_adherence")

    graph.add_conditional_edges(
        "judge_adherence",
        adherence_router,
        {"adheres": "write_code_to_file", "refine": "adherence_refinement"},
    )
    graph.add_edge("adherence_refinement", "judge_adherence")

    graph.add_edge("write_code_to_file", "execute_manim")

    graph.add_conditional_edges(
        "execute_manim", execute_router, {"success": END, "refine": "error_refinement"}
    )
    graph.add_edge("error_refinement", "write_code_to_file")

    return graph.compile()


def main():
    handler = TerminalHandler()
    # Default to Grok; can be changed to "gpt-4o"
    model = os.getenv("MODEL_NAME", "grok")
    llm_provider = LLMProvider(model_name=model)

    graph = build_graph(handler, llm_provider)
    state = AppState(user_query="")

    while True:
        state = graph.invoke(state)
        if state is None:  # User quit
            break
        if state["video_path"]:
            handler.display_output(f"Video generated at: {state["video_path"]}")
        elif state["error"]:
            handler.display_error(state["error"])
        else:
            handler.display_error("Unexpected state: No video or error.")
        # Reset state for next query
        state = AppState(user_query="")


if __name__ == "__main__":
    main()
