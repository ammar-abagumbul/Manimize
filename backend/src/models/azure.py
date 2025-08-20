from base import BaseLLM
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

class AzureAI(BaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.credential),
            model=self.model
        )

    def complete(self, messages):
        messages_list = [dict(m) for m in messages]
        response = self.llm.complete(messages=messages_list)
        return response.choices[0].message.content
