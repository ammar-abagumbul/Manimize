from typing import Iterable
from .model_registry import MODEL_REGISTRY
from ..types.chat_completions import ChatCompletionMessageType


def get_model_client(model_name: str):
    """
    Retrieve the model client based on the model name.

    Args:
        model_name (str): The name of the model to retrieve.

    Returns:
        object: An instance of the model client.
    """
    config = MODEL_REGISTRY[model_name]
    return config.client_class(
        endpoint=config.endpoint,
        credential=config.token,
        model=config.model
    )


def call_llm(messages: Iterable[ChatCompletionMessageType], model_name: str = "openai") -> str:
    """
    Call the language model with a series of messages.

    Args:
        messages (Iterable[ChatCompletionMessageType]): The messages to send to the model.
        model_name (str): The name of the model to use. Defaults to "openai".

    Returns:
        str: The response from the language model.
    """
    client = get_model_client(model_name)
    return client.complete(messages=messages)
