import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
AZURE_API = os.getenv("AZURE_API")
OUTPUT_FILE = Path.cwd() / "generated_manim.py"
LAST_ERROR_STATE = False
LOG_FILE = "./stdout.log"
MANIM_FILE = "./generated_manim.py"

# Model configuration
OPENAI_MODEL = "openai/gpt-4.1"
OPENAI_ENDPOINT = "https://models.github.ai/inference"
TEMPERATURE = 0.0
