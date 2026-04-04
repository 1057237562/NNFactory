import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any
from code_generator import CodeGenerator
from training_engine import TrainingEngine

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

class TrainConfig(BaseModel):
    blueprint: Blueprint
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    scheduler: str = "none"
    weight_decay: float = 0.0
    step_size: int = 30
    gamma: float = 0.1
    input_size: list[int] = [3, 224, 224]
    num_classes: int = 10
    num_samples: int = 1000
    val_ratio: float = 0.2

class EvalConfig(BaseModel):
    blueprint: Blueprint
    input_size: list[int] = [3, 224, 224]
    num_classes: int = 10
    num_samples: int = 1000
    val_ratio: float = 0.2
    loss_function: str = "cross_entropy"

training_engines: dict[str, TrainingEngine] = {}

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

@app.post("/train")
async def train_model(config: TrainConfig):
    generator = CodeGenerator(config.blueprint)
    engine = TrainingEngine(config.blueprint, generator)
    train_id = f"train_{config.blueprint.model_name}"
    training_engines[train_id] = engine

    train_config = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "loss_function": config.loss_function,
        "scheduler": config.scheduler,
        "weight_decay": config.weight_decay,
        "step_size": config.step_size,
        "gamma": config.gamma,
        "input_size": config.input_size,
        "num_classes": config.num_classes,
        "num_samples": config.num_samples,
        "val_ratio": config.val_ratio
    }

    def event_stream():
        for event in engine.train(train_config):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/train/stop")
async def stop_training():
    for engine in training_engines.values():
        engine.stop_training()
    return {"status": "stopped"}

@app.post("/evaluate")
async def evaluate_model(config: EvalConfig):
    generator = CodeGenerator(config.blueprint)
    engine = TrainingEngine(config.blueprint, generator)

    eval_config = {
        "input_size": config.input_size,
        "num_classes": config.num_classes,
        "num_samples": config.num_samples,
        "val_ratio": config.val_ratio,
        "loss_function": config.loss_function
    }

    result = engine.evaluate(eval_config)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}
