import yaml
import uuid
import subprocess
import shutil

from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Iterable

from .model_registry import MODEL_REGISTRY
from ..types.chat_completions import ChatCompletionMessageType
from pocketflow import Node

load_dotenv()

model_name = "openai"

def get_model_client(model_name: str):
    config = MODEL_REGISTRY[model_name]
    return config.client_class(
        endpoint=config.endpoint,
        credential=config.token,
        model=config.model
    )

def call_llm(messages: Iterable[ChatCompletionMessageType]):
    client = get_model_client(model_name)
    return client.complete(messages=messages)

class StepLogger:
    def log_step(self, msg: str| None = None):
        class_name = self.__class__.__name__
        print(f"[{class_name}] - {"" if msg is None else msg}")


class GenerateOutlineNode(Node, StepLogger):

    def prep(self, shared: Dict):
        return shared.get("user_query", None)

    def exec(self, prep_res: str | None):
        self.log_step()
        query = prep_res
        system_prompt = "You are a skilled mathematician with the ability to explain complex mathematical concepts in a clear and egaging manner in the style of 3Blue1Brown. Your task is given a request, to generate a structured outline."
        user_prompt = "Generate a YAML outline for explaining '{concept}' in 3Blue1Brown style." \
                      "**Example Outline**" \
                      "```yaml" \
                      "topic: Introduction to Prime Numbers" \
                      "sections: " \
                      " - title: Understanding Prime Numbers" \
                      "   description: Explain what prime numbers are, their definition (natural numbers greater than 1 divisible only by 1 and themselves), and their significance in mathematics, including examples like 2, 3, 5, and 7." \
                      "   visuals: " \
                      "     - Animated number line highlighting prime numbers up to 100." \
                      "     - Visual comparison of prime vs. composite numbers using factor trees." \
                      "```"

        response = call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(concept=query)}
        ])
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        yaml_data = yaml.safe_load(yaml_str)
        return yaml_data

    def exec_fallback(self, prep_res, exc):
        print("An error occured", str(exc))
        return "Failed gracefully"

    def post(self, shared, prep_res, exec_res):
        shared["outline"] = exec_res
        return "default"


class ManimCodeGenerationNode(Node, StepLogger):
    def prep(self, shared):
        return shared.get("outline", "")

    def exec(self, prep_res):
        self.log_step()
        outline = prep_res
        system_prompt = "You are a skilled mathematician and Manim animator. Your task is to convert an outline that a user provides to a full fledged Manim code that is ready to be executed."
        user_prompt = "Generate a Manim code for the following outline: {outline}. You must use `from manim import *` for your import. You must strictly follow the following output format" \
                      "Make sure to name your scene ExplanationScene" \
                      "**Example Output**" \
                      "```python" \
                      "from manim import *" \
                      "class ExplanationScene(Scene):" \
                      "    def construct(self):" \
                      "        circle = Circle()" \
                      "        self.play(Create(circle))" \
                      "```"
        response = call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(outline=outline)}
        ])
        print(f"[Generated manim code]\n: {response}")
        code = response.split("```python")[1].split("```")[0].strip()
        return code

    def exec_fallback(self, prep_res, exc):
        print("An error occured", str(exc))

    def post(self, shared, prep_res, exec_res):
        shared['manim_code'] = exec_res
        return "default"

class QualityChecker(Node, StepLogger):
    """Checks whether the generated manim code closely adheres to the outline specified in previous nodes."""
    def prep(self, shared):
        return shared.get("outline", ""), shared.get("manim_code", "")

    def exec(self, prep_res):
        self.log_step()
        outline, manim_code = prep_res
        system_prompt = "You are an objective agent whose job is to make judgement whether a given manim code adheres to the outline that was used to generate the code."
        user_prompt = "Given the following outline:\n {outline} and the following manim code:\n {manim_code}, your task is to return a short yaml response containing a verdict and a feedback" \
                      "A verdict can either be `proceed` or `revise`. If your verdict is `proceed`, you do not have to give any feedback." \
                      "**Example output**" \
                      "```yaml" \
                      "verdict: proceed" \
                      'feedback: "The code does not include animations for the number line as specified in the outline."'  \
                      "```"
        response = call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(outline=outline, manim_code=manim_code)}
        ])
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        yaml_data = yaml.safe_load(yaml_str)
        return yaml_data["verdict"], yaml_data["feedback"]

    def exec_fallback(self, prep_res, exc):
        print("An error occured", str(exc))
        return "Error"

    def post(self, shared, prep_res, exec_res):
        verdict = prep_res["verdict"]
        feedback = prep_res["feedback"]
        shared['verdict'] = verdict
        shared['feedback'] = feedback

        if verdict == "proceed":
            return "proceed"
        else:
            return "verdict"

class ManimExecutor(Node, StepLogger):
    """Executes manim"""
    def prep(self, shared):
        return shared.get("manim_code", "")

    def exec(self, prep_res):
        self.log_step()
        manim_code = prep_res["manim_code"]
        scripts_dir = Path("scripts")
        scripts_dir.mkdir(exist_ok=True)
        file_name = uuid.uuid4()
        file_path = scripts_dir / f"{file_name}.py"
        with open(file_path, "w") as f:
            f.write(manim_code)

        # execute manim code
        command = ["manim", "-pql", file_path, "ExplanationScene"]
        result = subprocess.run(command, capture_output=True, text=True)
        error = ""
        if result.returncode != 0:
            error = result.stderr

        video_path = (
            Path.cwd() / "media" / "videos" / str(file_name) / "480p15" / "ExplanationScene.mp4"
        )

        videos_dir = Path.cwd() / "videos"
        videos_dir.mkdir(exist_ok=True)
        if video_path.exists():
            target_video_path = videos_dir / video_path.name
            video_path.replace(target_video_path)
        else:
            target_video_path = None

        media_dir = Path.cwd() / "media"
        if media_dir.exists() and media_dir.is_dir():
            shutil.rmtree(media_dir)

        return str(file_path), str(target_video_path) if target_video_path else None, error

    def exec_fallback(self, prep_res, exc):
        print("An error occured", str(exc))
        return "Error"

    def post(self, shared, prep_res, exec_res):
        file, video_path, error = exec_res
        shared["error"] = error
        shared["file"]  = file
        shared["video_path"] = video_path
        if error:
            return "correct_error"
        else:
            return "default"

class ErrorResolver(Node, StepLogger):
    def prep(self, shared):
        return shared

    def exec(self, prep_res):
        code = prep_res["manim_code"]
        error = prep_res["error"]

        system_prompt = "You are a skilled mathematician, coder and manim animator. Your task is to correct the defective manim code without altering much of the content and information within."
        user_prompt = "Given the following manim code: " \
                      "```python" \
                      "{manim_code}" \
                      "```" \
                      "And the following error message that occurs upon executing the manim script: " \
                      "<Error> {error} </Error>" \
                      "apply correction to the manim code without altering the content or the information." \
                      "Your response should strictly follow the following output format. " \
                      "**Example output**" \
                      "```python" \
                      "from manim import *" \
                      "class ExplanationScene(Scene):" \
                      "    def construct(self):" \
                      "        circle = Circle()" \
                      "        self.play(Create(circle))" \
                      "```"

        response = call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(manim_code=code, error=error)}
        ])
        code = response.split("```python")[1].split("```")[0].strip()
        return code

    def exec_fallback(self, prep_res, exc):
        print("Error:", str(exc))
        return "Error"

    def post(self, shared, prep_res, exec_res):
        shared["manim_code"] = exec_res
        return "default"

class END(Node):
    pass
