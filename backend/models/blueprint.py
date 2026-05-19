from pydantic import BaseModel
from typing import Any


class LayerConfig(BaseModel):
    id: str
    type: str
    params: dict[str, Any]
    position: dict[str, float]


class Connection(BaseModel):
    from_id: str
    to_id: str


class Blueprint(BaseModel):
    layers: list[LayerConfig]
    connections: list[Connection]
    model_name: str = "NeuralNetwork"
    use_jit: bool = False
    use_compile: bool = False
    device: str = "cpu"
