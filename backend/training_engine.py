import importlib.util
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image


class DeviceDataLoader:
    def __init__(self, loader: DataLoader[Any], device: str) -> None:
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.startswith("cuda") else None
        self._iterator = None

    def __iter__(self) -> Any:
        if self.stream is None:
            for batch in self.loader:
                yield tuple(t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in batch)
            return

        self._iterator = iter(self.loader)
        try:
            batch = next(self._iterator)
            batch = tuple(t.pin_memory() if isinstance(t, torch.Tensor) else t for t in batch)
        except StopIteration:
            return

        for next_batch in self._iterator:
            with torch.cuda.stream(self.stream):
                next_batch = tuple(
                    t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                    for t in next_batch
                )

            yield tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in batch
            )

            torch.cuda.current_stream().synchronize()
            batch = next_batch

        with torch.cuda.stream(self.stream):
            batch = tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in batch
            )
        torch.cuda.current_stream().synchronize()
        yield batch

    def __len__(self) -> int:
        return len(self.loader)


class ImageFolderWithTransform(torch.utils.data.Dataset[Any]):
    def __init__(self, root_dir: str, transform: Optional[Callable[[Image.Image], Any]] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if self.classes:
            for cls_name in self.classes:
                cls_path = os.path.join(root_dir, cls_name)
                for f in os.listdir(cls_path):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")):
                        self.samples.append((os.path.join(cls_path, f), self.class_to_idx[cls_name]))
        else:
            for f in os.listdir(root_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")):
                    self.samples.append((os.path.join(root_dir, f), 0))
            self.classes = ["default"]
            self.class_to_idx = {"default": 0}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TrainingEngine:
    TEMP_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")

    def __init__(self, blueprint: Any, code_generator: Any) -> None:
        self.blueprint = blueprint
        self.code_generator = code_generator
        self.is_training = False
        self._stop_event = threading.Event()
        self._temp_path: Optional[str] = None
        self._weights_path: Optional[str] = None
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

    def _cleanup(self) -> None:
        self._temp_path = None
        self._weights_path = None

    def _create_synthetic_dataset(self, config: dict[str, Any]) -> tuple[DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]], int]:
        input_size = config.get("input_size", [3, 224, 224])
        num_classes = config.get("num_classes", 10)
        num_samples = config.get("num_samples", 1000)
        val_ratio = config.get("val_ratio", 0.2)
        batch_size = config.get("batch_size", 32)

        train_size = int(num_samples * (1 - val_ratio))
        val_size = num_samples - train_size
        c = input_size[0]
        h = input_size[1] if len(input_size) > 1 else 1
        w = input_size[2] if len(input_size) > 2 else 1

        x_train = torch.randn(train_size, c, h, w)
        y_train = torch.randint(0, num_classes, (train_size,))
        x_val = torch.randn(val_size, c, h, w)
        y_val = torch.randint(0, num_classes, (val_size,))

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, pin_memory=True)
        return train_loader, val_loader, num_classes

    def _create_dataset_from_config(self, config: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any], int]:
        dataset_id: Any = config.get("dataset_id")
        if dataset_id:
            from dataset_manager import DatasetManager  # pyright: ignore[reportImplicitRelativeImport]
            dm = DatasetManager()
            ds_info: Optional[dict[str, Any]] = dm.get_dataset(dataset_id)
            if ds_info:
                return self._load_real_dataset(ds_info, config)
        return self._create_synthetic_dataset(config)

    @staticmethod
    def _build_image_data_loaders(
        data_path: str, transform: Any, batch_size: int, val_ratio: float
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        full_dataset = ImageFolderWithTransform(data_path, transform=transform)
        train_size = int(len(full_dataset) * (1 - val_ratio))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return train_loader, val_loader

    def _load_real_dataset(self, ds_info: dict[str, Any], config: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any], int]:
        batch_size = config.get("batch_size", 32)
        val_ratio = config.get("val_ratio", 0.2)
        input_size = ds_info.get("input_shape", [3, 224, 224])
        num_classes = ds_info.get("num_classes", 10)
        dataset_type = ds_info.get("dataset_type", "")
        data_path = ds_info.get("file_path", "")

        if dataset_type == "image_classification":
            normalize: transforms.Normalize
            if input_size[0] == 3:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                normalize = transforms.Normalize(mean=[0.5], std=[0.5])
            transform = transforms.Compose([
                transforms.Resize((input_size[1], input_size[2])),
                transforms.ToTensor(),
                normalize,
            ])
            train_loader, val_loader = self._build_image_data_loaders(
                data_path, transform, batch_size, val_ratio
            )
            return train_loader, val_loader, num_classes

        elif dataset_type == "image_folder":
            transform = transforms.Compose([
                transforms.Resize((input_size[1], input_size[2])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            train_loader, val_loader = self._build_image_data_loaders(
                data_path, transform, batch_size, val_ratio
            )
            return train_loader, val_loader, num_classes

        elif dataset_type == "tabular_csv":
            import csv
            import numpy as np
            feature_cols = ds_info.get("metadata", {}).get("feature_columns", [])
            label_col = ds_info.get("metadata", {}).get("label_column")
            numeric_cols = ds_info.get("metadata", {}).get("numeric_columns", [])

            with open(data_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if label_col and rows and label_col in rows[0]:
                labels_str = [row[label_col] for row in rows]
                unique_labels = sorted(set(labels_str))
                label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
                y = np.array([label_map[lbl] for lbl in labels_str])
                num_classes = len(unique_labels)
            else:
                y = np.zeros(len(rows))
                num_classes = 0

            cols_to_use = numeric_cols if numeric_cols else feature_cols
            x_matrix = np.zeros((len(rows), len(cols_to_use)))
            for i, row in enumerate(rows):
                for j, col in enumerate(cols_to_use):
                    try:
                        x_matrix[i, j] = float(row.get(col, 0))
                    except (ValueError, TypeError):
                        x_matrix[i, j] = 0

            mean = x_matrix.mean(axis=0)
            std = x_matrix.std(axis=0) + 1e-8
            x_normalized = (x_matrix - mean) / std

            x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

            dataset = TensorDataset(x_tensor, y_tensor)
            train_size = int(len(dataset) * (1 - val_ratio))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            return train_loader, val_loader, num_classes

        else:
            return self._create_synthetic_dataset(config)

    def train(self, config: dict[str, Any]) -> Any:
        self.is_training = True
        self._stop_event.clear()

        try:
            device = config.get("device", "cpu")
            requested_device = device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            device_obj = torch.device(device)
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

            device_info = f"{device_obj} ({torch.cuda.get_device_name(0)})" if device == "cuda" else "cpu"
            if requested_device == "cuda" and device == "cpu":
                device_info += " [CUDA unavailable, fell back to CPU]"

            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
            epochs = int(config.get("epochs", 10))
            total_steps = epochs * len(train_loader)
            step_count = 0
            start_time = time.time()

            yield {
                "type": "device_info",
                "device": device_info,
                "cuda_available": torch.cuda.is_available(),
                "requested": requested_device,
                "actual": device
            }

            if self._stop_event.is_set():
                yield {"type": "stopped", "message": "Training cancelled", "epochs_completed": 0, "total_epochs": epochs}
                return

            for epoch in range(epochs):
                if self._stop_event.is_set():
                    break

                epoch_loss, epoch_acc, step_count, progress_events = self._train_one_epoch(
                    model, train_loader, criterion, optimizer, device, epoch, epochs, total_steps, step_count, start_time
                )

                for evt in progress_events:
                    yield evt

                if self._stop_event.is_set():
                    break

                val_loss, val_acc = self._evaluate_model(model, val_loader, criterion, device)

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
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            model = self._build_model(device)
            _, val_loader, num_classes = self._create_dataset_from_config(config)
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
        # Standard classification: output (N, C) vs target (N,)
        if output.dim() == target.dim() + 1:
            return criterion(output, target)
        # Same shape: regression (MSELoss), binary with matching dims
        if output.shape == target.shape:
            return criterion(output, target.float())
        # Fallback: flatten output to (N, -1) and let criterion handle it
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
        optimizers: dict[str, Callable[[], optim.Optimizer]] = {
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
