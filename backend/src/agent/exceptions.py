class WorkflowError(Exception):
    """
    Base exception for workflow-related errors.
    """
    pass


class LLMError(WorkflowError):
    """
    Exception raised for errors related to the language model.
    """
    def __init__(self, message: str):
        super().__init__(f"LLM Error: {message}")


class ManimExecutionError(WorkflowError):
    """
    Exception raised for errors during Manim script execution.
    """
    def __init__(self, message: str):
        super().__init__(f"Manim Execution Error: {message}")