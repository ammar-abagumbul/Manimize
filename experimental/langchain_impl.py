# %%

import os
import json
from typing import Dict, Annotated
from pathlib import Path
import subprocess

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool
from langchain_core.tools import ToolException
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.globals import set_verbose, set_debug

from dotenv import load_dotenv

from pydantic import BaseModel, Field

load_dotenv()

AZURE_API = os.getenv("AZURE_API")

# $$

OUTPUT_FILE = Path.cwd() / "generated_manim.py"
LAST_ERROR_STATE = False
LOG_FILE = "./stdout.log"
MANIM_FILE = "./generated_manim.py"

manim_file = Path(MANIM_FILE)

# %%


class StructuredManimCode(BaseModel):
    code: str = Field(
        description="The generated Manim code",
    )


# %%


def extract_structured_output(response: str) -> Dict[str, str]:
    try:
        response = json.loads(response.content)
        return response
    except json.JSONDecodeError:
        raise ValueError("Response content is not valid JSON.")


@tool
def write_to_file(
    code: Annotated[str, "manim code to be written"],
) -> None:
    """Useful to write a valid and executable manim code into a designated python file."""
    try:
        manim_file.write_text(code)
    except Exception as e:
        return f"Error writing to file: {e}"


# %%


@tool
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


# %%


@tool
def get_execution_output() -> str:
    """Get latest manim execution stdout and stderr. Only to be used in case latest execution fails with stderr"""
    log_file = Path("./stdout.log")
    return log_file.read_text()


# %%


@tool
def get_error_status():
    """Check the error status of latest Manim code execution."""
    return LAST_ERROR_STATE


# %%

parser = PydanticOutputParser(pydantic_object=StructuredManimCode, format="json")

# %%

openai = "openai/gpt-4.1"
llama = "meta/Llama-4-Scout-17B-16E-Instruct"

# %%

model = AzureAIChatCompletionsModel(
    endpoint="https://models.github.ai/inference",
    credential=AZURE_API,  # if using Entra ID you can should use DefaultAzureCredential() instead
    model=openai,
    temperature=0,
)

# %%

# template = """You are a helpful assistant that helps users visualize mathematical concepts and problems using manim. You have access to the following tools: {tools}.
# When a user asks you a question, you shall only respond by a valid manim code. When you generate a manim code, make sure you first write into a file before attempting to execute the code. You shall not include any explanation outside the code. The path to the files you have access to are not provided to you, but you may expect the tools to achieve their tasks regardless.

template = """You are a helpful assistant that helps users visualize mathematical concepts and problems using manim and then write the manim code to a file. After writing the code, you will execute it using the manim tool. If the execution fails, you will retrieve the error output, access the cause of the error and generate a new code which fixes the issue.
You have access to the following tools: {tools}.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 5 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# %%

set_verbose(True)
# set_debug(True)

tools = [write_to_file, run_manim_code, get_execution_output]

agent = create_react_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke(
    {"input": "What are the different types of triangles."}, verbose=True
)


# %%
