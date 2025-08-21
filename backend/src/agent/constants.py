# Constants for workflow_pocketflow

# Prompts for the language model
GENERATE_OUTLINE_PROMPT = (
    "You are a skilled mathematician with the ability to explain complex mathematical concepts in a clear and "
    "engaging manner in the style of 3Blue1Brown. Your task is given a request, to generate a structured outline."
)

MANIM_CODE_GENERATION_PROMPT = (
    "You are a skilled mathematician and Manim animator. Your task is to convert an outline that a user provides "
    "to a full-fledged Manim code that is ready to be executed. "
    "Make sure that you are using the star import, i.e `from manim import *`"
)

QUALITY_CHECK_PROMPT = (
    "You are an objective agent whose job is to make a judgment on whether a given Manim code adheres to the outline "
    "that was used to generate the code."
)

ERROR_RESOLUTION_PROMPT = (
    "You are a skilled mathematician, coder, and Manim animator. Your task is to correct the defective Manim code "
    "without altering much of the content and information within."
)

# Example outputs for prompts
EXAMPLE_OUTLINE = (
    "**Example Output**"
    "```yaml\n"
    "topic: Introduction to Prime Numbers\n"
    "sections:\n"
    "  - title: Understanding Prime Numbers\n"
    "    description: Explain what prime numbers are, their definition (natural numbers greater than 1 divisible only by 1 and themselves), and their significance in mathematics, including examples like 2, 3, 5, and 7.\n"
    "    visuals:\n"
    "      - Animated number line highlighting prime numbers up to 100.\n"
    "      - Visual comparison of prime vs. composite numbers using factor trees.\n"
    "```"
)

EXAMPLE_MANIM_CODE = (
    "**Example Output**\n"
    "```python\n"
    "from manim import *\n"
    "class ExplanationScene(Scene):\n"
    "    def construct(self):\n"
    "        circle = Circle()\n"
    "        self.play(Create(circle))\n"
    "```"
)

EXAMPLE_YAML_QUALITY = (
    "**Example Output**\n"
    "```yaml\n"
    "verdict: proceed \n"
    'feedback: "The code does not include animations for the number line as specified in the outline."'
)
