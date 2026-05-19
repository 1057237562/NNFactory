import importlib.util
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Optional, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..preprocessing_metadata import PreprocessingMetadata
from ..device_utils import (
    resolve_device,
    get_device_name,
    is_cuda_available,
    is_rocm_available,
    is_xpu_available,
    is_mps_available,
)
from .dataloader import (
    DeviceDataLoader,
    _create_synthetic_dataset,
    _create_dataset_from_config,
    _build_image_data_loaders,
    _create_gpu_resident_loaders,
    _load_real_dataset,
)


class TrainingEngine:
    TEMP_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")

    # Data-loading methods imported from dataloader module
    _create_synthetic_dataset = _create_synthetic_dataset
    _create_dataset_from_config = _create_dataset_from_config
    _build_image_data_loaders = staticmethod(_build_image_data_loaders)
    _create_gpu_resident_loaders = _create_gpu_resident_loaders
    _load_real_dataset = _load_real_dataset

    def __init__(self, blueprint: Any, code_generator: Any) -> None:
        self.blueprint = blueprint
        self.code_generator = code_generator
        self.is_training = False
        self._stop_event = threading.Event()
        self._temp_path: Optional[str] = None
        self._weights_path: Optional[str] = None
        self._preprocessing_meta: Optional[PreprocessingMetadata] = None
        os.makedirs(self.TEMP_DIR, exist_ok=True)

    def _build_model(self, device: str = "cpu") -> Any:
        code = self.code_generator.generate()
        self._temp_path = self._write_temp_module(code)
        module = self._load_module(self._temp_path)
        model_class = getattr(module, self.blueprint.model_name)
        model = model_class()
        model = model.to(device)
        return model

    def _write_temp_module(self, code: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.blueprint.model_name}_{timestamp}.py"
        path = os.path.join(self.TEMP_DIR, filename)
        with open(path, "w") as f:
            f.write(code)
        return path

    @staticmethod
    def _load_module(path: str) -> Any:
        spec = importlib.util.spec_from_file_location("_nnfactory_model", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _prune_old_checkpoints(model_name: str, keep: int = 3) -> None:
        temp_dir = TrainingEngine.TEMP_DIR
        if not os.path.isdir(temp_dir):
            return

        prefix = f"{model_name}_"
        checkpoints = [
            f for f in os.listdir(temp_dir)
            if f.endswith(".pth") and f.startswith(prefix)
        ]
        checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(temp_dir, f)), reverse=True)

        for old in checkpoints[keep:]:
            old_path = os.path.join(temp_dir, old)
            try:
                os.remove(old_path)
            except OSError:
                pass
            meta_path = old_path + ".meta.json"
            if os.path.exists(meta_path):
                try:
                    os.remove(meta_path)
                except OSError:
                    pass

    def _cleanup(self) -> None:
        self._temp_path = None
        self._weights_path = None
        self._preprocessing_meta = None

    @staticmethod
    def _check_gpu_memory(tensor_size_bytes: int, device: str = "cuda") -> bool:
        if device == "xpu":
            try:
                if not torch.xpu.is_available():
                    return False
                total = torch.xpu.get_device_properties(0).total_memory
                allocated = torch.xpu.memory_allocated(0)
                available = total - allocated
                return tensor_size_bytes * 2 < available
            except Exception:
                return False
        elif device == "mps":
            try:
                return torch.backends.mps.is_available()
            except Exception:
                return False
        else:
            if not torch.cuda.is_available():
                return False
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                available = total - allocated
                return tensor_size_bytes * 2 < available
            except Exception:
                return False

    def train(self, config: dict[str, Any]) -> Any:
        self.is_training = True
        self._stop_event.clear()

        try:
            device = config.get("device", "cpu")
            requested_device = device
            resolved = resolve_device(device)
            target_unavailable = (
                (resolved == "cuda" and not torch.cuda.is_available()) or
                (resolved == "xpu" and not is_xpu_available()) or
                (resolved == "mps" and not is_mps_available())
            )
            if target_unavailable:
                device = "cpu"
            device_obj = torch.device(device)
            torch.set_num_threads(config.get("num_threads", 4))
            if self._stop_event.is_set():
                yield {"type": "stopped", "message": "Training cancelled", "epochs_completed": 0}
                return
            model = self._build_model(device)
            if self._stop_event.is_set():
                yield {"type": "stopped", "message": "Training cancelled", "epochs_completed": 0}
                return
            train_loader, val_loader, num_classes = self._create_dataset_from_config(config)
            if self._stop_event.is_set():
                yield {"type": "stopped", "message": "Training cancelled", "epochs_completed": 0}
                return

            is_gpu_resident = callable(train_loader)
            train_factory: Any = None
            val_factory: Any = None
            if is_gpu_resident:
                train_factory = train_loader
                val_factory = val_loader
            else:
                train_loader = DeviceDataLoader(train_loader, device)
                val_loader = DeviceDataLoader(val_loader, device)
            criterion = self._get_criterion(str(config.get("loss_function", "cross_entropy"))).to(device_obj)
            optimizer = self._get_optimizer(
                str(config.get("optimizer", "adam")), model,
                float(config.get("learning_rate", 0.001)),
                float(config.get("weight_decay", 0.0))
            )
            scheduler = self._get_scheduler(
                str(config.get("scheduler", "none")), optimizer,
                int(config.get("step_size", 30)),
                float(config.get("gamma", 0.1))
            )

            device_name = get_device_name(requested_device)
            device_info = f"{device_obj} ({device_name})" if device != "cpu" else "cpu"
            if requested_device != "cpu" and device == "cpu":
                device_info += f" [{requested_device.upper()} unavailable, fell back to CPU]"

            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
            epochs = int(config.get("epochs", 10))
            if is_gpu_resident:
                n = config.get("num_samples", 0)
                batch_size = config.get("batch_size", 32)
                val_ratio = config.get("val_ratio", 0.2)
                if n > 0:
                    train_samples = int(n * (1 - val_ratio))
                    total_steps = epochs * max(1, train_samples // batch_size)
                else:
                    total_steps = epochs * 100
            else:
                total_steps = epochs * len(cast(DeviceDataLoader, train_loader))
            step_count = 0
            start_time = time.time()

            yield {
                "type": "device_info",
                "device": device_info,
                "device_type": requested_device,
                "cuda_available": torch.cuda.is_available(),
                "rocm_available": is_rocm_available(),
                "xpu_available": is_xpu_available(),
                "mps_available": is_mps_available(),
                "requested": requested_device,
                "actual": device
            }

            if self._stop_event.is_set():
                yield {"type": "stopped", "message": "Training cancelled", "epochs_completed": 0, "total_epochs": epochs}
                return

            epoch = 0
            for epoch in range(epochs):
                if self._stop_event.is_set():
                    break

                if is_gpu_resident:
                    epoch_train_loader = train_factory()
                    epoch_val_loader = val_factory()
                else:
                    epoch_train_loader = train_loader
                    epoch_val_loader = val_loader

                epoch_loss, epoch_acc, step_count, progress_events = self._train_one_epoch(
                    model, epoch_train_loader, criterion, optimizer, device, epoch, epochs, total_steps, step_count, start_time
                )

                for evt in progress_events:
                    yield evt

                if self._stop_event.is_set():
                    break

                val_loss, val_acc = self._evaluate_model(model, epoch_val_loader, criterion, device)

                if scheduler is not None and str(config.get("scheduler", "none")) != "none":
                    if config.get("scheduler") == "reduce_on_plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                history["train_loss"].append(epoch_loss)
                history["val_loss"].append(val_loss)
                history["train_acc"].append(epoch_acc)
                history["val_acc"].append(val_acc)
                history["lr"].append(float(optimizer.param_groups[0]["lr"]))

                yield {
                    "type": "epoch_end",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "train_acc": epoch_acc,
                    "val_acc": val_acc,
                    "progress": ((epoch + 1) / epochs) * 100,
                    "elapsed": time.time() - start_time,
                    "history": history
                }

            if self._stop_event.is_set():
                yield {
                    "type": "stopped",
                    "epochs_completed": epoch + 1,
                    "total_epochs": epochs,
                    "total_time": time.time() - start_time,
                    "history": history,
                    "message": "Training stopped by user"
                }
                return

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            weights_filename = f"{self.blueprint.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            self._weights_path = os.path.join(self.TEMP_DIR, weights_filename)
            torch.save(model.state_dict(), self._weights_path)

            if self._preprocessing_meta is not None:
                self._preprocessing_meta.save(self._weights_path)

            self._prune_old_checkpoints(self.blueprint.model_name, keep=3)

            yield {
                "type": "complete",
                "epochs_completed": epoch + 1,
                "total_epochs": epochs,
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "final_train_acc": history["train_acc"][-1],
                "final_val_acc": history["val_acc"][-1],
                "total_params": total_params,
                "trainable_params": trainable_params,
                "total_time": time.time() - start_time,
                "history": history,
                "weights_path": weights_filename
            }

        except Exception as e:
            yield {"type": "error", "message": f"Training failed: {str(e)}", "traceback": traceback.format_exc()}
        finally:
            self.is_training = False
            self._cleanup()

    def _train_one_epoch(
        self,
        model: Any,
        train_loader: Any,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
        epoch: int,
        total_epochs: int,
        total_steps: int,
        step_count: int,
        start_time: float
    ) -> tuple[float, float, int, list[dict[str, Any]]]:
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        progress_events = []

        for batch_x, batch_y in train_loader:
            if self._stop_event.is_set():
                break

            optimizer.zero_grad()
            output = self._forward(model, batch_x)
            loss = self._compute_loss(output, batch_y, criterion)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_y.size(0)
            train_total += batch_y.size(0)
            if output.dim() > 1:
                train_correct += int(output.max(1)[1].eq(batch_y).sum().item())
            else:
                train_correct += int((output > 0).long().eq(batch_y).sum().item())

            step_count += 1
            if step_count % max(1, total_steps // 50) == 0:
                progress_events.append({
                    "type": "progress",
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "step": step_count,
                    "total_steps": total_steps,
                    "progress": min((step_count / total_steps) * 100, 100),
                    "elapsed": time.time() - start_time,
                    "train_loss": train_loss / max(train_total, 1),
                    "train_acc": train_correct / max(train_total, 1) * 100
                })

        return train_loss / max(train_total, 1), train_correct / max(train_total, 1) * 100, step_count, progress_events

    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        try:
            device = config.get("device", "cpu")
            resolved = resolve_device(device)
            if (resolved == "cuda" and not torch.cuda.is_available()) or \
               (resolved == "xpu" and not is_xpu_available()) or \
               (resolved == "mps" and not is_mps_available()):
                device = "cpu"
            torch.set_num_threads(config.get("num_threads", 4))
            model = self._build_model(device)
            _, val_loader, num_classes = self._create_dataset_from_config(config)
            if callable(val_loader):
                val_loader = val_loader()
            else:
                val_loader = DeviceDataLoader(val_loader, device)
            criterion = self._get_criterion(str(config.get("loss_function", "cross_entropy"))).to(device)
            val_loss, val_acc = self._evaluate_model(model, val_loader, criterion, device)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            per_class = self._compute_per_class_stats(model, val_loader, num_classes, device)

            return {
                "status": "success",
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "num_classes": num_classes,
                "per_class_accuracy": per_class
            }
        except Exception as e:
            return {"status": "error", "message": f"Evaluation failed: {str(e)}", "traceback": traceback.format_exc()}
        finally:
            self._cleanup()

    def _evaluate_model(self, model: Any, data_loader: Any, criterion: nn.Module, device: str = "cpu") -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                output = self._forward(model, batch_x)
                loss = self._compute_loss(output, batch_y, criterion)
                total_loss += loss.item() * batch_y.size(0)
                total += batch_y.size(0)
                if output.dim() > 1:
                    correct += int(output.max(1)[1].eq(batch_y).sum().item())
                else:
                    correct += int((output > 0).long().eq(batch_y).sum().item())

        return total_loss / max(total, 1), correct / max(total, 1) * 100

    def _compute_per_class_stats(
        self, model: Any, data_loader: Any, num_classes: int, device: str = "cpu"
    ) -> list[dict[str, Any]]:
        model.eval()
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                output = self._forward(model, batch_x)
                if output.dim() > 1:
                    predicted = output.max(1)[1]
                    for i in range(batch_y.size(0)):
                        label = int(batch_y[i].item())
                        if label < num_classes:
                            class_total[label] += 1
                            if int(predicted[i].item()) == label:
                                class_correct[label] += 1

        return [
            {"class": i, "accuracy": (class_correct[i] / max(class_total[i], 1)) * 100, "samples": class_total[i]}
            for i in range(min(num_classes, 10))
        ]

    def stop_training(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _forward(model: Any, x: Any) -> torch.Tensor:
        output = model(x)
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _compute_loss(output: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        if output.dim() == 1:
            output = output.unsqueeze(-1)
        if output.dim() == target.dim() + 1:
            return criterion(output, target)
        if output.shape == target.shape:
            return criterion(output, target.float())
        output = output.reshape(output.size(0), -1)
        return criterion(output, target)

    @staticmethod
    def _get_criterion(name: str) -> nn.Module:
        criteria: dict[str, nn.Module] = {
            "cross_entropy": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss(),
            "bce": nn.BCEWithLogitsLoss(),
            "l1": nn.L1Loss(),
            "nll": nn.NLLLoss(),
        }
        return criteria.get(name, nn.CrossEntropyLoss())

    @staticmethod
    def _get_optimizer(name: str, model: Any, lr: float, weight_decay: float) -> optim.Optimizer:
        optimizers: dict[str, Any] = {
            "adam": lambda: optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
            "adamw": lambda: optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
            "sgd": lambda: optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay),
            "rmsprop": lambda: optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay),
            "adagrad": lambda: optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay),
        }
        return optimizers.get(name, lambda: optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))()

    @staticmethod
    def _get_scheduler(name: str, optimizer: optim.Optimizer, step_size: int, gamma: float) -> Optional[Any]:
        schedulers: dict[str, Any] = {
            "step_lr": optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
            "cosine": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size),
            "exponential": optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma),
            "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=5),
        }
        return schedulers.get(name)
