# utils.py
# Utility functions for the Manimizer application

import os
import re
import subprocess
from datetime import datetime

def generate_filename(topic: str, extension: str = ".py") -> str:
    """Generate a unique filename based on the concept topic and current timestamp."""
    clean_topic = re.sub(r'[^a-zA-Z0-9]', '_', topic).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{clean_topic}_{timestamp}{extension}"

def ensure_directory(directory: str) -> None:
    """Ensure the specified directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)

def write_to_file(content: str, filepath: str) -> bool:
    """Write content to a file, ensuring the directory exists."""
    directory = os.path.dirname(filepath)
    if directory:
        ensure_directory(directory)
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing to file {filepath}: {e}")
        return False

def execute_manim_script(filepath: str, output_dir: str, quality: str = "low") -> dict:
    """Execute a Manim script and return the result."""
    quality_flags = {
        "low": "-pql",
        "medium": "-pqm",
        "high": "-pqh"
    }
    quality_flag = quality_flags.get(quality, "-pql")
    
    command = f"manim {quality_flag} {filepath} -o {output_dir}"
    try:
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        if process.returncode == 0:
            # Extract video path from output if possible, or assume standard location
            video_path = os.path.join(output_dir, os.path.basename(filepath).replace('.py', '.mp4'))
            return {"success": True, "video_path": video_path, "error_message": None}
        else:
            return {"success": False, "video_path": None, "error_message": process.stderr}
    except Exception as e:
        return {"success": False, "video_path": None, "error_message": str(e)}
