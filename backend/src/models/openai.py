from base import BaseLLM
from openai import OpenAI

class OpenAILLM(BaseLLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = OpenAI(
            base_url=self.endpoint,
            api_key=self.credential
        )

    def complete(self, messages):
        if self.model is None:
            self.model = "openai/gpt-4o" # default

        completion = self.llm.chat.completions.create(
            messages=messages,
            model=self.model
        )

        response = completion.choices[0].message.content
        return response if response else ""
