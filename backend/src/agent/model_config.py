from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    endpoint: str
    token: str
    model: str
    client_class: type
