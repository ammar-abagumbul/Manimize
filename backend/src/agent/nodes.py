import yaml
import logging
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Tuple

from pocketflow import Node
from .utils import call_llm
from .constants import (
    EXAMPLE_YAML_QUALITY,
    GENERATE_OUTLINE_PROMPT,
    MANIM_CODE_GENERATION_PROMPT,
    QUALITY_CHECK_PROMPT,
    ERROR_RESOLUTION_PROMPT,
    EXAMPLE_OUTLINE,
    EXAMPLE_MANIM_CODE,
)
from .exceptions import LLMError, ManimExecutionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StepLogger:
    def log_step(self, msg: str = ""):
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] - {msg}")


class GenerateOutlineNode(Node, StepLogger):
    def prep(self, shared: Dict):
        return shared.get("user_query", None)

    def exec(self, prep_res: str | None):
        self.log_step("Generating outline")
        if not prep_res:
            raise ValueError("User query is missing.")

        response = call_llm([
            {"role": "system", "content": GENERATE_OUTLINE_PROMPT},
            {"role": "user", "content": f"Generate a YAML outline for explaining '{prep_res}' in 3Blue1Brown style. Follow the following outline strictly. \n{EXAMPLE_OUTLINE}"},
        ])

        try:
            yaml_str = response.split("```yaml")[1].split("```")[0].strip()
            return yaml.safe_load(yaml_str)
        except Exception as e:
            raise LLMError(f"Failed to parse YAML response: {e}")

    def post(self, shared, prep_res, exec_res):
        shared["outline"] = exec_res
        logger.info(shared)
        return "default"


class ManimCodeGenerationNode(Node, StepLogger):
    def prep(self, shared):
        return shared.get("outline", "")

    def exec(self, prep_res):
        self.log_step("Generating Manim code")
        if not prep_res:
            raise ValueError("Outline is missing.")

        response = call_llm([
            {"role": "system", "content": MANIM_CODE_GENERATION_PROMPT},
            {"role": "user", "content": f"Generate a Manim code for the following outline:\n{prep_res}\n Your output should match the following format. {EXAMPLE_MANIM_CODE}"},
        ])

        try:
            code = response.split("```python")[1].split("```")[0].strip()
            return code
        except Exception as e:
            raise LLMError(f"Failed to parse Python code: {e}")

    def post(self, shared, prep_res, exec_res):
        shared["manim_code"] = exec_res
        logger.info(shared)
        return "default"


class QualityChecker(Node, StepLogger):
    def prep(self, shared):
        return shared.get("outline", ""), shared.get("manim_code", "")

    def exec(self, prep_res: Tuple[str, str]):
        self.log_step("Checking quality of Manim code")
        outline, manim_code = prep_res
        if not outline or not manim_code:
            raise ValueError("Outline or Manim code is missing.")

        response = call_llm([
            {"role": "system", "content": QUALITY_CHECK_PROMPT},
            {"role": "user", "content": f"Given the following outline:\n{outline}\nand the following Manim code:\n{manim_code}\nProvide a YAML response with a verdict and feedback. Follow the following output format. {EXAMPLE_YAML_QUALITY}"},
        ])

        try:
            yaml_str = response.split("```yaml")[1].split("```")[0].strip()
            yaml_data = yaml.safe_load(yaml_str)
            return yaml_data["verdict"], yaml_data.get("feedback", "")
        except Exception as e:
            raise LLMError(f"Failed to parse YAML response: {e}")

    def post(self, shared, prep_res, exec_res):
        verdict, feedback = exec_res
        shared["verdict"] = verdict
        shared["feedback"] = feedback
        logger.info(shared)
        return "proceed" if verdict == "proceed" else "revise"


class ManimExecutor(Node, StepLogger):
    def prep(self, shared):
        return shared.get("manim_code", "")

    def exec(self, prep_res):
        self.log_step("Executing Manim code")
        if not prep_res:
            raise ValueError("Manim code is missing.")

        scripts_dir = Path("scripts")
        scripts_dir.mkdir(exist_ok=True)
        file_name = uuid.uuid4()
        file_path = scripts_dir / f"{file_name}.py"

        with open(file_path, "w") as f:
            f.write(prep_res)

        command = ["manim", "-pql", str(file_path), "ExplanationScene"]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise ManimExecutionError(result.stderr)

        video_path = (
            Path.cwd() / "media" / "videos" / str(file_name) / "480p15" / "ExplanationScene.mp4"
        )

        if video_path.exists():
            target_video_path = Path("videos") / video_path.name
            target_video_path.parent.mkdir(exist_ok=True)
            video_path.replace(target_video_path)
            return str(file_path), str(target_video_path)
        else:
            raise FileNotFoundError("Generated video not found.")

    def post(self, shared, prep_res, exec_res):
        file, video_path = exec_res
        shared["file"] = file
        shared["video_path"] = video_path
        logger.info(shared)
        return "default"


class ErrorResolver(Node, StepLogger):
    def prep(self, shared):
        return shared

    def exec(self, prep_res):
        self.log_step("Resolving errors in Manim code")
        code = prep_res.get("manim_code", "")
        error = prep_res.get("error", "")

        if not code or not error:
            raise ValueError("Manim code or error message is missing.")

        response = call_llm([
            {"role": "system", "content": ERROR_RESOLUTION_PROMPT},
            {"role": "user", "content": f"Given the following Manim code:\n```python\n{code}\n```\nand the following error:\n<Error> {error} </Error>\nApply corrections."},
        ])

        try:
            corrected_code = response.split("```python")[1].split("```")[0].strip()
            return corrected_code
        except Exception as e:
            raise LLMError(f"Failed to parse corrected Python code: {e}")

    def post(self, shared, prep_res, exec_res):
        shared["manim_code"] = exec_res
        logger.info(shared)
        return "default"
