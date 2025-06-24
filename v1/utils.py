import subprocess
from pathlib import Path
from typing_extensions import Annotated

from config import MANIM_FILE, LOG_FILE

manim_file = Path(MANIM_FILE)

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
    log_file = Path(LOG_FILE)
    return log_file.read_text()
