from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from code_generator import CodeGenerator

app = FastAPI(title="NNFactory Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/generate")
async def generate_code(blueprint: Blueprint):
    generator = CodeGenerator(blueprint)
    code = generator.generate()
    return {"code": code, "status": "success"}

@app.post("/validate")
async def validate_blueprint(blueprint: Blueprint):
    try:
        generator = CodeGenerator(blueprint)
        generator.validate()
        return {"valid": True, "errors": []}
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}

@app.get("/health")
async def health():
    return {"status": "ok"}
