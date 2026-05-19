import os
from datetime import datetime
from typing import Any

import csv
import hashlib
import time

from .operations import (
    PreprocessingResult,
    _apply_remove_samples,
    _apply_balance,
    _apply_normalize,
    _apply_one_hot,
    _apply_label_encode,
    _apply_ordinal_encode,
    _apply_target_encode,
    _apply_frequency_encode,
    _apply_binary_encode,
    _apply_hash_encode,
    _create_temp_dataset,
    _export_dataset,
)


class PreprocessingPipeline:
    # Operation methods imported from operations module
    _apply_remove_samples = _apply_remove_samples
    _apply_balance = _apply_balance
    _apply_normalize = _apply_normalize
    _apply_one_hot = _apply_one_hot
    _apply_label_encode = _apply_label_encode
    _apply_ordinal_encode = _apply_ordinal_encode
    _apply_target_encode = _apply_target_encode
    _apply_frequency_encode = _apply_frequency_encode
    _apply_binary_encode = _apply_binary_encode
    _apply_hash_encode = _apply_hash_encode
    _create_temp_dataset = _create_temp_dataset
    _export_dataset = _export_dataset

    def _apply_filter_class(self, params):
        classes_str = params.get("classes", "")
        mode = params.get("mode", "keep")

        if not classes_str:
            classes = []
        else:
            classes = [c.strip() for c in classes_str.split(",") if c.strip()]

        if self.dataset_type == "tabular_csv":
            label_col = self.metadata.get("label_column")
            if not label_col:
                return {"affected_samples": 0, "affected_columns": 0}

            filtered_rows = []
            affected = 0
            with open(self.file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                for row in reader:
                    row_label = row.get(label_col, "")
                    in_classes = row_label in classes if classes else True
                    should_keep = in_classes if mode == "keep" else not in_classes

                    if should_keep:
                        filtered_rows.append(row)
                    else:
                        affected += 1

            self.num_samples = len(filtered_rows)
            temp_path = self._create_temp_dataset(headers, filtered_rows)
            self.file_path = temp_path

            return {
                "affected_samples": affected,
                "affected_columns": 0,
                "message": f"Filtered {affected} samples ({mode} mode)"
            }

        elif self.dataset_type in ("image_classification", "image_folder"):
            return {"affected_samples": 0, "affected_columns": 0}

        return {"affected_samples": 0, "affected_columns": 0}

    def _apply_split(self, params):
        train_ratio = params.get("train_ratio", 0.8)
        val_ratio = params.get("val_ratio", 0.2)

        self.metadata["split_info"] = {
            "train": train_ratio,
            "val": val_ratio
        }

        return {"affected_samples": 0, "affected_columns": 0}

    def _apply_resize(self, params):
        width = params.get("width", 224)
        height = params.get("height", 224)

        if self.dataset_type in ("image_classification", "image_folder"):
            self.metadata["resize_config"] = {"width": width, "height": height}

            return {
                "affected_samples": self.num_samples,
                "affected_columns": 0,
                "message": f"Resize configured to {width}x{height} (image transformation on dataloader)"
            }

        return {"affected_samples": 0, "affected_columns": 0}

    def __init__(self, source_dataset_id: str, dataset_manager):
        self.source_id = source_dataset_id
        self.ds_manager = dataset_manager
        self.source_info = self.ds_manager.get_dataset(source_dataset_id)
        if not self.source_info:
            raise ValueError(f"Dataset {self.source_id} not found")

        self.dataset_type = self.source_info.get("dataset_type", "")
        self.file_path = self.source_info.get("file_path", "")
        self.num_samples = self.source_info.get("num_samples", 0)
        self.num_classes = self.source_info.get("num_classes", 0)
        self.metadata = self.source_info.get("metadata", {})
        self.transformations_applied: list = []
        self.affected_samples = 0
        self.affected_columns = 0

    def execute(self, operations: list[dict]) -> PreprocessingResult:
        try:
            for op in operations:
                op_type = op.get("type")
                params = op.get("params", {})

                if not op_type:
                    continue

                if op_type == "split":
                    self._apply_split(params)
                    continue

                if op_type == "purge_all":
                    continue

                method = getattr(self, f"_apply_{op_type}", None)
                if method:
                    result = method(params)
                    self.affected_samples += result.get("affected_samples", 0)
                    self.affected_columns += result.get("affected_columns", 0)
                    self.transformations_applied.append({
                        "type": op_type,
                        "params": params
                    })
                else:
                    raise ValueError(f"Unknown operation: {op_type}")

            return self._export_dataset()

        except Exception as e:
            return PreprocessingResult(
                success=False,
                message=f"Preprocessing failed: {str(e)}",
                errors=[str(e)]
            )
