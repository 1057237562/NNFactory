import json
import os
import zipfile
import tempfile
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import Any, Optional
from .code_generator import CodeGenerator
from .models.blueprint import Blueprint
from .training.engine import TrainingEngine
from .datasets.manager import DatasetManager
from .preprocessing.pipeline import PreprocessingPipeline
from .preprocessing.operations import PreprocessingResult
from .device_utils import is_cuda_available, is_rocm_available, is_xpu_available, is_mps_available
from .evaluator import CustomEvaluator

app = FastAPI(title="NNFactory Backend", version="1.0.0")
executor = ThreadPoolExecutor(max_workers=4)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    num_threads: int = 4
    val_ratio: float = 0.2

class EvalConfig(BaseModel):
    blueprint: Blueprint
    input_size: list[int] = [3, 224, 224]
    num_classes: int = 10
    num_samples: int = 1000
    val_ratio: float = 0.2
    loss_function: str = "cross_entropy"
    device: str = "cpu"
    unknown_strategy: str = "error"

class SingleRowInput(BaseModel):
    blueprint: Blueprint
    features: dict[str, str]
    weights_filename: Optional[str] = None
    unknown_strategy: str = "error"

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
    num_threads: int = 4
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
        "num_threads": config.num_threads,
        "device": config.blueprint.device
    }

    async def event_stream():
        loop = asyncio.get_event_loop()
        train_executor = ThreadPoolExecutor(max_workers=1)
        iterator = iter(engine.train(train_config))
        try:
            while True:
                try:
                    event = await loop.run_in_executor(train_executor, next, iterator)
                    yield f"data: {json.dumps(event)}\n\n"
                except StopIteration:
                    break
        finally:
            train_executor.shutdown(wait=False)
            training_engines.pop(train_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/train/stop")
async def stop_training():
    for engine in training_engines.values():
        engine.stop_training()
    return {"status": "stopped"}

@app.post("/evaluate/synthetic")
async def evaluate_synthetic(config: EvalConfig):
    generator = CodeGenerator(config.blueprint)
    engine = TrainingEngine(config.blueprint, generator)

    eval_config = {
        "input_size": config.input_size,
        "num_classes": config.num_classes,
        "num_samples": config.num_samples,
        "val_ratio": config.val_ratio,
        "loss_function": config.loss_function,
        "device": config.device,
    }

    result = engine.evaluate(eval_config)
    return result

@app.post("/evaluate/detect-type")
async def detect_model_type(blueprint: Blueprint):
    try:
        evaluator = CustomEvaluator(blueprint)
        result = evaluator.detect_type()
        weights_dir = CustomEvaluator.TEMP_DIR
        weights_available = False
        weights_list = []
        if os.path.exists(weights_dir):
            for f in sorted(os.listdir(weights_dir)):
                if f.endswith(".pth"):
                    weights_available = True
                    path = os.path.join(weights_dir, f)
                    weights_list.append({
                        "filename": f,
                        "size": os.path.getsize(path)
                    })
        return {
            **result,
            "weights_available": weights_available,
            "weights_list": weights_list
        }
    except ValueError as e:
        return {"type": "unknown", "input_shape": [], "num_classes": 0, "valid": False, "error": str(e)}

@app.post("/evaluate/image")
async def evaluate_images(
    blueprint: str = Form(...),
    weights_filename: Optional[str] = Form(None),
    top_k: int = Form(5),
    images: list[UploadFile] = File(...),
):
    if len(images) > 50:
        return {"valid": False, "errors": ["Maximum 50 images per request"]}

    try:
        bp = Blueprint(**json.loads(blueprint))
    except Exception as e:
        return {"valid": False, "errors": [f"Invalid blueprint: {str(e)}"]}

    evaluator = CustomEvaluator(bp)

    if weights_filename:
        weights_path = os.path.join(CustomEvaluator.TEMP_DIR, weights_filename)
        result = evaluator.load_weights(weights_path)
        if not result.get("valid", True):
            return result

    temp_dir = tempfile.mkdtemp()
    try:
        image_paths = []
        image_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        for img in images:
            content = await img.read()
            filename = img.filename or "image.png"
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            if ext == '.zip':
                zip_path = os.path.join(temp_dir, filename)
                with open(zip_path, "wb") as f:
                    f.write(content)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for entry in zf.namelist():
                        entry_lower = entry.lower()
                        if any(entry_lower.endswith(e) for e in image_exts):
                            extracted = os.path.join(temp_dir, os.path.basename(entry))
                            os.makedirs(os.path.dirname(extracted), exist_ok=True)
                            with zf.open(entry) as src, open(extracted, 'wb') as dst:
                                dst.write(src.read())
                            image_paths.append(extracted)
                os.remove(zip_path)
            else:
                path = os.path.join(temp_dir, filename)
                with open(path, "wb") as f:
                    f.write(content)
                image_paths.append(path)

        if len(image_paths) > 50:
            return {"valid": False, "errors": ["Maximum 50 images per request (after ZIP extraction)"]}

        result = evaluator.evaluate_images(image_paths, top_k)
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/evaluate/tabular")
async def evaluate_tabular_csv(
    blueprint: str = Form(...),
    weights_filename: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.endswith(".csv"):
        return {"valid": False, "errors": ["File must be a CSV"]}

    try:
        bp = Blueprint(**json.loads(blueprint))
    except Exception as e:
        return {"valid": False, "errors": [f"Invalid blueprint: {str(e)}"]}

    evaluator = CustomEvaluator(bp)

    if weights_filename:
        weights_path = os.path.join(CustomEvaluator.TEMP_DIR, weights_filename)
        wresult = evaluator.load_weights(weights_path)
        if not wresult.get("valid", True):
            return wresult

    temp_dir = tempfile.mkdtemp()
    try:
        content = await file.read()
        csv_path = os.path.join(temp_dir, file.filename)
        with open(csv_path, "wb") as f:
            f.write(content)

        result = evaluator.evaluate_tabular_csv(csv_path)
        if not result.get("valid", True):
            return result

        with open(result["output_path"], "rb") as f:
            csv_content = f.read()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="eval_{file.filename}"'
            },
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/evaluate/tabular/single")
async def evaluate_tabular_single(input_data: SingleRowInput):
    try:
        evaluator = CustomEvaluator(input_data.blueprint)

        if input_data.weights_filename:
            weights_path = os.path.join(CustomEvaluator.TEMP_DIR, input_data.weights_filename)
            wresult = evaluator.load_weights(weights_path)
            if not wresult.get("valid", True):
                return wresult

        result = evaluator.evaluate_tabular_single(input_data.features, input_data.unknown_strategy)
        return result
    except ValueError as e:
        return {"valid": False, "errors": [str(e)]}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/devices")
async def list_devices():
    cuda_avail, cuda_vendor = is_cuda_available()
    return {
        "devices": [
            {"id": "cpu",  "name": "CPU",                "available": True},
            {"id": "cuda", "name": "NVIDIA CUDA",         "available": cuda_avail and cuda_vendor == "nvidia"},
            {"id": "rocm", "name": "AMD ROCm",            "available": is_rocm_available()},
            {"id": "xpu",  "name": "Intel XPU",           "available": is_xpu_available()},
            {"id": "mps",  "name": "Apple MPS",           "available": is_mps_available()},
        ]
    }

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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, dataset_manager.get_visualization, dataset_id)

@app.get("/datasets/{dataset_id}/column-stats")
async def column_stats(dataset_id: str, column: Optional[str] = Query(None)):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, dataset_manager.get_column_stats, dataset_id, column)

@app.get("/datasets/{dataset_id}/config")
async def get_dataloader_config(dataset_id: str):
    return dataset_manager.get_dataloader_config(dataset_id)

@app.get("/datasets/{dataset_id}/target-columns")
async def get_target_columns(dataset_id: str):
    return dataset_manager.get_available_target_columns(dataset_id)

@app.put("/datasets/{dataset_id}/label-column")
async def set_label_column(dataset_id: str, label_column: str = Query(...)):
    return dataset_manager.set_label_column(dataset_id, label_column)

@app.post("/datasets/{dataset_id}/preprocess")
async def preprocess_dataset(dataset_id: str, operations: list[dict[str, Any]]):
    try:
        pipeline = PreprocessingPipeline(dataset_id, dataset_manager)
        result = pipeline.execute(operations)
        
        if result.success:
            return {
                "valid": True,
                "message": result.message,
                "affected_samples": result.affected_samples,
                "affected_columns": result.affected_columns,
                "new_dataset_id": result.new_dataset_id
            }
        else:
            return {
                "valid": False,
                "errors": result.errors
            }
    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)]
        }

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
        "num_threads": config.num_threads,
        "dataset_id": config.dataset_id,
        "device": config.blueprint.device,
    }

    def _next_or_none(it):
        try:
            return next(it)
        except StopIteration:
            return None

    async def event_stream():
        loop = asyncio.get_event_loop()
        train_executor = ThreadPoolExecutor(max_workers=1)
        iterator = iter(engine.train(train_config))
        try:
            while True:
                event = await loop.run_in_executor(train_executor, _next_or_none, iterator)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            train_executor.shutdown(wait=False)
            training_engines.pop(train_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/weights/{filename}")
async def download_weights(filename: str):
    weights_dir = TrainingEngine.TEMP_DIR
    file_path = os.path.join(weights_dir, filename)
    if not os.path.exists(file_path):
        return {"valid": False, "errors": ["Weights file not found"]}
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

@app.get("/weights")
async def list_weights():
    weights_dir = TrainingEngine.TEMP_DIR
    if not os.path.exists(weights_dir):
        return {"weights": []}
    weights = []
    for f in sorted(os.listdir(weights_dir)):
        if f.endswith(".pth"):
            path = os.path.join(weights_dir, f)
            stat = os.stat(path)
            weights.append({
                "filename": f,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "size_human": _format_size(stat.st_size)
            })
    return {"weights": weights}

def _format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

@app.delete("/weights/{filename}")
async def delete_weights(filename: str):
    weights_dir = TrainingEngine.TEMP_DIR
    file_path = os.path.join(weights_dir, filename)
    if not os.path.exists(file_path):
        return {"status": "not_found", "message": "Weights file not found"}
    os.remove(file_path)
    return {"status": "deleted", "filename": filename}

@app.post("/weights/purge")
async def purge_weights():
    weights_dir = TrainingEngine.TEMP_DIR
    if not os.path.exists(weights_dir):
        return {"status": "purged", "count": 0}
    count = 0
    for f in os.listdir(weights_dir):
        if f.endswith(".pth"):
            os.remove(os.path.join(weights_dir, f))
            count += 1
    return {"status": "purged", "count": count}
