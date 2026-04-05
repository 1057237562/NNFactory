import json
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Optional
from code_generator import CodeGenerator
from training_engine import TrainingEngine
from dataset_manager import DatasetManager

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

class TrainWithDatasetConfig(BaseModel):
    blueprint: Blueprint
    dataset_id: str
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    scheduler: str = "none"
    weight_decay: float = 0.0
    step_size: int = 30
    gamma: float = 0.1
    val_ratio: float = 0.2

training_engines: dict[str, TrainingEngine] = {}
dataset_manager = DatasetManager()

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
        "val_ratio": config.val_ratio,
        "device": config.blueprint.device
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

@app.get("/datasets")
async def list_datasets():
    return {"datasets": dataset_manager.list_datasets()}

@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    result = dataset_manager.get_dataset(dataset_id)
    if result is None:
        return {"valid": False, "errors": ["Dataset not found"]}
    return result

@app.post("/datasets/upload")
async def upload_dataset(
    file: Optional[UploadFile] = File(None),
    name: Optional[str] = Form(None),
    label_column: Optional[str] = Form(None),
    source_path: Optional[str] = Form(None),
):
    if source_path:
        result = dataset_manager.load_from_folder(source_path, name)
        return result

    if file is None:
        return {"valid": False, "errors": ["No file or path provided"]}

    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    temp_dir = tempfile.mkdtemp()
    try:
        content = await file.read()
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, "wb") as f:
            f.write(content)

        if ext == ".csv":
            result = dataset_manager.load_from_csv(temp_path, name, label_column)
        elif ext in (".zip",):
            import zipfile
            extract_dir = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(temp_path, "r") as zf:
                zf.extractall(extract_dir)
            top_level = os.listdir(extract_dir)
            if len(top_level) == 1:
                target = os.path.join(extract_dir, top_level[0])
            else:
                target = extract_dir
            result = dataset_manager.load_from_folder(target, name)
        else:
            result = {"valid": False, "errors": [f"Unsupported file type: {ext}. Use .csv or a folder/.zip of images."]}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return result

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    return dataset_manager.delete_dataset(dataset_id)

@app.post("/datasets/purge")
async def purge_datasets():
    return dataset_manager.purge_all()

@app.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = Query(10, ge=1, le=50)):
    return dataset_manager.get_preview(dataset_id, limit)

@app.get("/datasets/{dataset_id}/visualize")
async def visualize_dataset(dataset_id: str):
    return dataset_manager.get_visualization(dataset_id)

@app.get("/datasets/{dataset_id}/column-stats")
async def column_stats(dataset_id: str, column: Optional[str] = Query(None)):
    return dataset_manager.get_column_stats(dataset_id, column)

@app.get("/datasets/{dataset_id}/config")
async def get_dataloader_config(dataset_id: str):
    return dataset_manager.get_dataloader_config(dataset_id)

@app.post("/train/dataset")
async def train_with_dataset(config: TrainWithDatasetConfig):
    generator = CodeGenerator(config.blueprint)
    engine = TrainingEngine(config.blueprint, generator)
    train_id = f"train_{config.blueprint.model_name}"
    training_engines[train_id] = engine

    ds_info = dataset_manager.get_dataset(config.dataset_id)
    if ds_info is None or not ds_info.get("valid", True):
        return {"valid": False, "errors": ["Dataset not found"]}

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
        "input_size": ds_info.get("input_shape", [3, 224, 224]),
        "num_classes": ds_info.get("num_classes", 10),
        "num_samples": ds_info.get("num_samples", 1000),
        "val_ratio": config.val_ratio,
        "dataset_id": config.dataset_id,
        "device": config.blueprint.device,
    }

    def event_stream():
        for event in engine.train(train_config):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
