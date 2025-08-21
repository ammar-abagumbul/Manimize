import os
from dotenv import load_dotenv

from .model_config import ModelConfig
from ..models import AzureAI, OpenAILLM

load_dotenv()

MODEL_REGISTRY = {
    "openai": ModelConfig(
        name="openai",
        endpoint="https://openrouter.ai/api/v1",
        token=os.environ["OPENROUTER_API"],
        model="openai/gpt-oss-120b",
        client_class=OpenAILLM,
    ),
    "azure": ModelConfig(
        name="azure",
        endpoint="https://azure.endpoint",
        token=os.environ["GITHUB_TOKEN"],
        model="azure/gpt-4",
        client_class=AzureAI,
    ),
}
