# manim_executor.py
# Functions for writing and executing Manim code

from ..core.models import ManimCode, ExecutionResult
from ..core.utils import generate_filename, write_to_file, execute_manim_script
from ..core.config import OUTPUT_SCRIPTS_DIR, OUTPUT_VIDEOS_DIR, MANIM_QUALITY

def write_and_execute_manim_code(manim_code: ManimCode, model) -> ExecutionResult:
    """Write Manim code to a file and execute it to generate a video."""
    # Generate a unique filename based on some identifier or timestamp
    filename = generate_filename("manim_script")  # Placeholder topic, can be improved with concept topic
    filepath = f"{OUTPUT_SCRIPTS_DIR}/{filename}"
    
    # Write the code to file
    success = write_to_file(manim_code.code, filepath)
    if not success:
        return ExecutionResult(success=False, error_message="Failed to write Manim code to file.")
    
    # Execute the Manim script
    result = execute_manim_script(filepath, OUTPUT_VIDEOS_DIR, MANIM_QUALITY)
    return ExecutionResult(
        success=result["success"],
        video_path=result["video_path"],
        error_message=result["error_message"]
    )
