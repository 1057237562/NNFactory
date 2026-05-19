import os
import csv
import hashlib
import time
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from PIL import Image

from ..preprocessing.metadata import PreprocessingMetadata
from ..device_utils import resolve_device


class DeviceDataLoader:
    def __init__(self, loader: DataLoader[Any], device: str) -> None:
        self.loader = loader
        self.device = device
        self.device_type = resolve_device(device)
        if self.device_type == "cuda":
            self.stream = torch.cuda.Stream()
        elif self.device_type == "xpu":
            try:
                self.stream = torch.xpu.Stream()
            except (AttributeError, RuntimeError):
                self.stream = None
        else:
            self.stream = None
        self._iterator = None

    def __iter__(self) -> Any:
        if self.stream is None:
            for batch in self.loader:
                yield tuple(t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in batch)
            return

        is_xpu = self.device_type == "xpu"
        stream_cm = torch.xpu.stream if is_xpu else torch.cuda.stream
        sync_fn = (torch.xpu.current_stream().synchronize
                   if is_xpu else torch.cuda.current_stream().synchronize)

        self._iterator = iter(self.loader)
        try:
            batch = next(self._iterator)
            batch = tuple(t.pin_memory() if isinstance(t, torch.Tensor) else t for t in batch)
        except StopIteration:
            return

        for next_batch in self._iterator:
            with stream_cm(cast(Any, self.stream)):
                next_batch = tuple(
                    t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                    for t in next_batch
                )

            yield tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in batch
            )

            batch = next_batch

        with stream_cm(cast(Any, self.stream)):
            batch = tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in batch
            )
        sync_fn()
        yield batch

    def __len__(self) -> int:
        return len(self.loader)


class _LenWrapper:
    """Wraps an iterable with a fixed __len__."""
    def __init__(self, iterable: Any, length: int) -> None:
        self._iterable = iterable
        self._length = length
    def __iter__(self) -> Any:
        return iter(self._iterable)
    def __len__(self) -> int:
        return self._length


class ImageFolderWithTransform(Dataset[Any]):
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


# ---------------------------------------------------------------------------
# Data-loading helper functions (assigned as methods on TrainingEngine)
# ---------------------------------------------------------------------------

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


def _create_dataset_from_config(self, config: dict[str, Any]) -> tuple[Any, Any, int]:
    dataset_id: Any = config.get("dataset_id")
    if dataset_id:
        from ..datasets.manager import DatasetManager
        dm = DatasetManager()
        ds_info: Optional[dict[str, Any]] = dm.get_dataset(dataset_id)
        if ds_info:
            return self._load_real_dataset(ds_info, config, config.get("device", "cpu"))
    return self._create_synthetic_dataset(config)


def _build_image_data_loaders(
    data_path: str, transform: Any, batch_size: int, val_ratio: float
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    full_dataset = ImageFolderWithTransform(data_path, transform=transform)
    train_size = int(len(full_dataset) * (1 - val_ratio))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    cpu_count = os.cpu_count() or 4
    n_workers = min(4, cpu_count // 4) if cpu_count >= 4 else 1

    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
        for _ in train_loader:
            break
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
    except RuntimeError:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def _create_gpu_resident_loaders(
    self,
    ds_info: dict[str, Any],
    config: dict[str, Any],
    device: str
) -> tuple[Any, Any, int]:
    import csv
    import numpy as np

    batch_size = config.get("batch_size", 32)
    val_ratio = config.get("val_ratio", 0.2)
    feature_cols = ds_info.get("metadata", {}).get("feature_columns", [])
    label_col = ds_info.get("metadata", {}).get("label_column")
    numeric_cols = ds_info.get("metadata", {}).get("numeric_columns", [])
    data_path = ds_info.get("file_path", "")

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

    meta = ds_info.get("metadata", {})
    is_normalized = meta.get("is_normalized", False)
    if not is_normalized:
        mean = x_matrix.mean(axis=0)
        std = x_matrix.std(axis=0) + 1e-8
        x_normalized = (x_matrix - mean) / std
    else:
        x_normalized = x_matrix
        mean = None
        std = None

    self._preprocessing_meta = PreprocessingMetadata(
        dataset_type="tabular_csv",
        dataset_id=ds_info.get("id", ""),
        feature_columns=feature_cols,
        numeric_columns=numeric_cols,
        label_column=label_col,
        is_normalized=is_normalized,
        normalization_mean=mean.tolist() if mean is not None else [],
        normalization_std=std.tolist() if std is not None else [],
        input_shape=[len(cols_to_use)],
        num_classes=num_classes,
        class_names=ds_info.get("class_names", []),
        one_hot_encoders=meta.get("one_hot_encoders", {}),
        label_encoders=meta.get("label_encoders", {}),
        ordinal_encoders=meta.get("ordinal_encoders", {}),
        target_encoders=meta.get("target_encoders", {}),
        frequency_encoders=meta.get("frequency_encoders", {}),
        binary_encoders=meta.get("binary_encoders", {}),
        hash_encoders=meta.get("hash_encoders", {}),
    )

    x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    bytes_needed = (
        x_tensor.nelement() * x_tensor.element_size()
        + y_tensor.nelement() * y_tensor.element_size()
    )
    gpu_ok = self._check_gpu_memory(bytes_needed, device)

    n = x_tensor.size(0)
    train_size = int(n * (1 - val_ratio))

    if gpu_ok:
        x_gpu = x_tensor.to(device)
        y_gpu = y_tensor.to(device)

        def _gpu_batch_generator(
            x: torch.Tensor, y: torch.Tensor, batch_sz: int, shuffle: bool = True
        ) -> Any:
            n_total = x.size(0)
            indices = torch.randperm(n_total) if shuffle else torch.arange(n_total)
            for i in range(0, n_total, batch_sz):
                idx = indices[i:i + batch_sz]
                yield x[idx], y[idx]

        def train_iter() -> Any:
            yield from _gpu_batch_generator(
                x_gpu[:train_size], y_gpu[:train_size], batch_size, shuffle=True
            )

        def val_iter() -> Any:
            yield from _gpu_batch_generator(
                x_gpu[train_size:], y_gpu[train_size:], batch_size, shuffle=False
            )

        return train_iter, val_iter, num_classes

    dataset = TensorDataset(x_tensor, y_tensor)
    val_size = n - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, num_classes


def _load_real_dataset(self, ds_info: dict[str, Any], config: dict[str, Any], device: str = "cpu") -> tuple[Any, Any, int]:
    batch_size = config.get("batch_size", 32)
    val_ratio = config.get("val_ratio", 0.2)
    input_size = ds_info.get("input_shape", [3, 224, 224])
    num_classes = ds_info.get("num_classes", 10)
    dataset_type = ds_info.get("dataset_type", "")
    data_path = ds_info.get("file_path", "")

    if dataset_type == "image_classification":
        normalize: transforms.Normalize
        if input_size[0] == 3:
            img_mean = [0.485, 0.456, 0.406]
            img_std = [0.229, 0.224, 0.225]
        else:
            img_mean = [0.5]
            img_std = [0.5]
        normalize = transforms.Normalize(mean=img_mean, std=img_std)
        transform = transforms.Compose([
            transforms.Resize((input_size[1], input_size[2])),
            transforms.ToTensor(),
            normalize,
        ])
        train_loader, val_loader = _build_image_data_loaders(
            data_path, transform, batch_size, val_ratio
        )
        self._preprocessing_meta = PreprocessingMetadata(
            dataset_type="image_classification",
            dataset_id=ds_info.get("id", ""),
            input_shape=input_size,
            resize_height=input_size[1] if len(input_size) > 1 else 224,
            resize_width=input_size[2] if len(input_size) > 2 else 224,
            image_normalization_mean=img_mean,
            image_normalization_std=img_std,
            num_classes=num_classes,
            class_names=ds_info.get("class_names", []),
        )
        return train_loader, val_loader, num_classes

    elif dataset_type == "image_folder":
        transform = transforms.Compose([
            transforms.Resize((input_size[1], input_size[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        train_loader, val_loader = _build_image_data_loaders(
            data_path, transform, batch_size, val_ratio
        )
        self._preprocessing_meta = PreprocessingMetadata(
            dataset_type="image_folder",
            dataset_id=ds_info.get("id", ""),
            input_shape=input_size,
            resize_height=input_size[1] if len(input_size) > 1 else 224,
            resize_width=input_size[2] if len(input_size) > 2 else 224,
            image_normalization_mean=[0.5],
            image_normalization_std=[0.5],
            num_classes=num_classes,
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

        meta = ds_info.get("metadata", {})
        is_normalized = meta.get("is_normalized", False)
        if not is_normalized:
            mean = x_matrix.mean(axis=0)
            std = x_matrix.std(axis=0) + 1e-8
            x_normalized = (x_matrix - mean) / std
        else:
            x_normalized = x_matrix
            mean = None
            std = None

        self._preprocessing_meta = PreprocessingMetadata(
            dataset_type="tabular_csv",
            dataset_id=ds_info.get("id", ""),
            feature_columns=feature_cols,
            numeric_columns=numeric_cols,
            label_column=label_col,
            is_normalized=is_normalized,
            normalization_mean=mean.tolist() if mean is not None else [],
            normalization_std=std.tolist() if std is not None else [],
            input_shape=[len(cols_to_use)],
            num_classes=num_classes,
            class_names=ds_info.get("class_names", []),
            one_hot_encoders=meta.get("one_hot_encoders", {}),
            label_encoders=meta.get("label_encoders", {}),
            ordinal_encoders=meta.get("ordinal_encoders", {}),
            target_encoders=meta.get("target_encoders", {}),
            frequency_encoders=meta.get("frequency_encoders", {}),
            binary_encoders=meta.get("binary_encoders", {}),
            hash_encoders=meta.get("hash_encoders", {}),
        )

        x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        if resolve_device(device) in ("cuda", "xpu", "mps"):
            return self._create_gpu_resident_loaders(ds_info, config, device)

        dataset = TensorDataset(x_tensor, y_tensor)
        train_size = int(len(dataset) * (1 - val_ratio))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return train_loader, val_loader, num_classes

    else:
        return self._create_synthetic_dataset(config)
