import os
import subprocess

from pathlib import Path
from dotenv import load_dotenv

from typing_extensions import Annotated, Dict

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain.base_language import BaseLanguageModel

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph
from langgraph.graph import MessagesState
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

model = AzureAIChatCompletionsModel(
    model=openai,
    endpoint="https://models.github.ai/inference",
    credential=AZURE_API,
    temperature=0.0,
)


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


class State(MessagesState):
    pass


class ManimizerAgent:
    def __init__(self, model: BaseLanguageModel, config: Dict[str, any] = None):
        self.sys_message = SystemMessage(
            content="""You are a Manimizer agent. When a user asks a question, you briefly explain the concept and then:
1. Generate the Manim code
2. Use write_to_file() to save it  
3. Use run_manim_code() with the class name to check if the written code runs without errros
4. Use get_execution_output() to check results
5. Debug if needed (max 3 attempts)

Always use the tools in this sequence. Your shall always generate a manim code and """
        )
        self.tools = [write_to_file, run_manim_code, get_execution_output]
        self.model = model.bind_tools(self.tools)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.config = config or {}

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MessagesState)
        workflow.add_node("generate_code_node", self.generate_code_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("generate_code_node")
        workflow.add_edge("tools", "generate_code_node")
        workflow.add_conditional_edges("generate_code_node", tools_condition)
        return workflow.compile(checkpointer=self.memory)

    def generate_code_node(self, state: State) -> Dict[str, any]:
        return {"messages": self.model.invoke([self.sys_message] + state["messages"])}

    def invoke(self, user_input: str):
        state = {"messages": [HumanMessage(content=user_input)]}
        return self.graph.invoke(state, self.config)

    def run_interaction(self) -> Dict[str, any]:
        print("Type 'exit' to stop the interaction.")
        user_input = input("User: ")
        state = {"messages": [HumanMessage(content=user_input)]}
        while user_input.lower() != "exit":
            state = self.invoke(user_input)
            # print the latest AIMessage
            last_message = state["messages"][-1].content
            print(f"AI: {last_message}")
            user_input = input("User: ")
        return state


manimizer_agent = ManimizerAgent(model, config={"configurable": {"thread_id": "1"}})
result = manimizer_agent.run_interaction()

for m in result["messages"]:
    print(m.pretty_print())
