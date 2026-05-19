"""Preprocessing metadata — captures how data was preprocessed during training
so the same transformations can be replayed during evaluation."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class PreprocessingMetadata:
    """Serialisable snapshot of the preprocessing applied during training.

    Saved as a ``.pth.meta.json`` sidecar file alongside trained weights.
    The evaluator loads this to replay identical transformations on new data.
    """

    # ---- Data identification ----
    dataset_type: str = ""  # "tabular_csv", "image_classification", "image_folder", "synthetic"
    dataset_id: str = ""

    # ---- Tabular: column selection ----
    feature_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    label_column: Optional[str] = None

    # ---- Tabular: normalisation stats (per-column) ----
    is_normalized: bool = False
    normalization_mean: list[float] = field(default_factory=list)
    normalization_std: list[float] = field(default_factory=list)

    # ---- Tabular: encoding maps ----
    one_hot_encoders: dict[str, Any] = field(default_factory=dict)
    label_encoders: dict[str, Any] = field(default_factory=dict)
    ordinal_encoders: dict[str, Any] = field(default_factory=dict)
    target_encoders: dict[str, Any] = field(default_factory=dict)
    frequency_encoders: dict[str, Any] = field(default_factory=dict)
    binary_encoders: dict[str, Any] = field(default_factory=dict)
    hash_encoders: dict[str, Any] = field(default_factory=dict)

    # ---- Image: spatial & colour normalisation ----
    input_shape: list[int] = field(default_factory=list)
    resize_height: int = 224
    resize_width: int = 224
    image_normalization_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_normalization_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # ---- Classification metadata ----
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)

    # ---- Free-form extras ----
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _meta_path(weights_path: str) -> str:
        return weights_path + ".meta.json"

    def save(self, weights_path: str) -> str:
        meta_path = self._meta_path(weights_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return meta_path

    @classmethod
    def load(cls, weights_path: str) -> Optional["PreprocessingMetadata"]:
        meta_path = cls._meta_path(weights_path)
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return None

    @classmethod
    def from_weights_filename(cls, weights_filename: str, temp_dir: str) -> Optional["PreprocessingMetadata"]:
        weights_path = os.path.join(temp_dir, weights_filename)
        return cls.load(weights_path)
