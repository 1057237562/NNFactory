import os
import json
import csv
import hashlib
import time
import numpy as np
from typing import Any, Optional, Sequence
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class PreprocessingResult:
    success: bool
    message: str
    affected_samples: int = 0
    affected_columns: int = 0
    new_dataset_id: Optional[str] = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_remove_samples(self, params):
    count = params.get("count", 100)
    strategy = params.get("strategy", "random")

    if self.dataset_type == "tabular_csv":
        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            all_rows = list(reader)

        if strategy == "first":
            removed = all_rows[:count]
            remaining = all_rows[count:]
        elif strategy == "last":
            removed = all_rows[-count:]
            remaining = all_rows[:-count]
        else:
            import random
            random.shuffle(all_rows)
            removed = all_rows[:count]
            remaining = all_rows[count:]

        self.num_samples = len(remaining)
        temp_path = self._create_temp_dataset(headers, remaining)
        self.file_path = temp_path

        return {
            "affected_samples": len(removed),
            "affected_columns": 0,
            "message": f"Removed {len(removed)} samples ({strategy} strategy)"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_balance(self, params):
    method = params.get("method", "undersample")

    if self.dataset_type == "tabular_csv":
        label_col = self.metadata.get("label_column")
        if not label_col:
            return {"affected_samples": 0, "affected_columns": 0}

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            all_rows = list(reader)

        by_label = {}
        for row in all_rows:
            lbl = row.get(label_col, "")
            if lbl not in by_label:
                by_label[lbl] = []
            by_label[lbl].append(row)

        max_size = max(len(rows) for rows in by_label.values()) if by_label else 0

        if method == "undersample":
            min_size = min(len(rows) for rows in by_label.values()) if by_label else 0
            balanced_rows = []
            affected = 0

            for lbl, rows in by_label.items():
                balanced_rows.extend(rows[:min_size])
                affected += len(rows) - min_size

            self.num_samples = len(balanced_rows)
            temp_path = self._create_temp_dataset(headers, balanced_rows)
            self.file_path = temp_path

            return {
                "affected_samples": affected,
                "affected_columns": 0,
                "message": f"Undersampled to {min_size} samples per class"
            }

        else:
            balanced_rows = []
            affected = 0

            for lbl, rows in by_label.items():
                while len(balanced_rows) + len(rows) <= max_size * len(by_label):
                    balanced_rows.extend(rows)
                    affected += len(rows)
                balanced_rows.extend(rows[:max_size - len(balanced_rows) % len(rows)])

            self.num_samples = len(balanced_rows)
            temp_path = self._create_temp_dataset(headers, balanced_rows)
            self.file_path = temp_path

            return {
                "affected_samples": affected,
                "affected_columns": 0,
                "message": f"Oversampled to {max_size} samples per class"
            }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_normalize(self, params):
    method = params.get("method", "zscore")

    if self.dataset_type == "tabular_csv":
        numeric_cols = self.metadata.get("numeric_columns", [])
        if not numeric_cols:
            return {"affected_samples": 0, "affected_columns": 0}

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            all_rows = list(reader)

        stats = {}
        for col in numeric_cols:
            values = []
            for row in all_rows:
                try:
                    values.append(float(row.get(col, 0) or 0))
                except (ValueError, TypeError):
                    values.append(0.0)

            if method == "zscore":
                mean = np.mean(values)
                std = np.std(values)
                stats[col] = {"mean": mean, "std": std, "min": np.min(values), "max": np.max(values)}

                for idx, row in enumerate(all_rows):
                    try:
                        val = values[idx]
                        normalized = (val - mean) / (std + 1e-8)
                        all_rows[idx][col] = str(round(normalized, 6))
                    except (ValueError, TypeError):
                        pass
            else:
                min_val = np.min(values)
                max_val = np.max(values)
                stats[col] = {"min": min_val, "max": max_val}

                for idx, row in enumerate(all_rows):
                    try:
                        val = values[idx]
                        normalized = (val - min_val) / (max_val - min_val + 1e-8)
                        all_rows[idx][col] = str(round(normalized, 6))
                    except (ValueError, TypeError):
                        pass

        temp_path = self._create_temp_dataset(headers, all_rows)
        self.file_path = temp_path

        if "normalization_stats" not in self.metadata:
            self.metadata["normalization_stats"] = {}
        self.metadata["normalization_stats"][method] = stats
        self.metadata["is_normalized"] = True

        return {
            "affected_samples": len(all_rows),
            "affected_columns": len(numeric_cols),
            "message": f"Normalized {len(numeric_cols)} columns via {method}"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_one_hot(self, params):
    columns_str = params.get("columns", "")
    drop_first = params.get("drop_first", False)
    max_categories = params.get("max_categories", 50)

    if self.dataset_type == "tabular_csv":
        if not columns_str:
            categorical_cols = [
                c for c in self.metadata.get("feature_columns", [])
                if c not in self.metadata.get("numeric_columns", [])
                and c != self.metadata.get("label_column")
            ]
        else:
            categorical_cols = [c.strip() for c in columns_str.split(",") if c.strip()]

        if not categorical_cols:
            return {"affected_samples": 0, "affected_columns": 0}

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            all_rows = list(reader)

        encoders = {}
        new_headers = list(headers)
        cols_to_remove = []
        cols_added = []

        for col in categorical_cols:
            if col not in all_rows[0]:
                continue

            values = sorted(set(row.get(col, "") for row in all_rows if row.get(col, "")))
            values = values[:max_categories]

            encoder_map = {v: i for i, v in enumerate(values)}
            encoders[col] = encoder_map

            if drop_first and len(values) > 1:
                values = values[1:]

            for i, val in enumerate(values):
                new_col = f"{col}_{val}"
                if new_col not in new_headers:
                    new_headers.append(new_col)
                cols_added.append(new_col)

                for row in all_rows:
                    row_val = row.get(col, "")
                    row[new_col] = "1" if row_val == val else "0"

            cols_to_remove.append(col)

        for col in cols_to_remove:
            new_headers.remove(col)
            for row in all_rows:
                row.pop(col, None)

        temp_path = self._create_temp_dataset(new_headers, all_rows)
        self.file_path = temp_path

        existing_feature_cols = [c for c in self.metadata.get("feature_columns", []) if c not in cols_to_remove]
        existing_feature_cols.extend(cols_added)
        self.metadata["feature_columns"] = existing_feature_cols

        existing_numeric = [c for c in self.metadata.get("numeric_columns", []) if c not in cols_to_remove]
        existing_numeric.extend(cols_added)
        self.metadata["numeric_columns"] = existing_numeric

        self.metadata["input_shape"] = [len(existing_numeric)] if existing_numeric else [len(existing_feature_cols)]

        if "one_hot_encoders" not in self.metadata:
            self.metadata["one_hot_encoders"] = {}
        self.metadata["one_hot_encoders"].update(encoders)

        return {
            "affected_samples": len(all_rows),
            "affected_columns": len(cols_added),
            "message": f"One-hot encoded {len(categorical_cols)} columns into {len(cols_added)} binary columns"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_label_encode(self, params):
    columns_str = params.get("columns", "")
    sort_by_freq = params.get("sort_by_freq", False)

    if self.dataset_type == "tabular_csv":
        if not columns_str:
            categorical_cols = [
                c for c in self.metadata.get("feature_columns", [])
                if c not in self.metadata.get("numeric_columns", [])
                and c != self.metadata.get("label_column")
            ]
        else:
            categorical_cols = [c.strip() for c in columns_str.split(",") if c.strip()]

        if not categorical_cols:
            return {"affected_samples": 0, "affected_columns": 0}

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            all_rows = list(reader)

        encoders = {}

        for col in categorical_cols:
            if col not in all_rows[0]:
                continue

            value_counts = {}
            for row in all_rows:
                val = row.get(col, "")
                value_counts[val] = value_counts.get(val, 0) + 1

            if sort_by_freq:
                sorted_values = sorted(value_counts.keys(), key=lambda v: value_counts[v], reverse=True)
            else:
                sorted_values = sorted(value_counts.keys())

            encoders[col] = {v: i for i, v in enumerate(sorted_values)}

            for row in all_rows:
                val = row.get(col, "")
                row[col] = str(encoders[col].get(val, 0))

        temp_path = self._create_temp_dataset(headers, all_rows)
        self.file_path = temp_path

        existing_numeric = [c for c in self.metadata.get("numeric_columns", []) if c not in categorical_cols]
        existing_numeric.extend(categorical_cols)
        self.metadata["numeric_columns"] = existing_numeric
        self.metadata["input_shape"] = [len(existing_numeric)]

        if "label_encoders" not in self.metadata:
            self.metadata["label_encoders"] = {}
        self.metadata["label_encoders"].update(encoders)

        return {
            "affected_samples": len(all_rows),
            "affected_columns": len(categorical_cols),
            "message": f"Label encoded {len(categorical_cols)} columns"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_ordinal_encode(self, params):
    columns_str = params.get("columns", "")
    mappings_str = params.get("mappings", "")

    if self.dataset_type == "tabular_csv":
        if not columns_str:
            categorical_cols = [
                c for c in self.metadata.get("feature_columns", [])
                if c not in self.metadata.get("numeric_columns", [])
                and c != self.metadata.get("label_column")
            ]
        else:
            categorical_cols = [c.strip() for c in columns_str.split(",") if c.strip()]

        custom_mappings = {}
        if mappings_str:
            try:
                custom_mappings = json.loads(mappings_str)
            except json.JSONDecodeError:
                pass

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            all_rows = list(reader)

        encoders = {}

        for col in categorical_cols:
            if col not in all_rows[0]:
                continue

            if col in custom_mappings and isinstance(custom_mappings[col], list):
                ordered_values = custom_mappings[col]
                remaining = [row.get(col, "") for row in all_rows if row.get(col, "") not in ordered_values]
                ordered_values.extend(remaining)
            else:
                values = sorted(set(row.get(col, "") for row in all_rows if row.get(col, "")))
                ordered_values = values

            encoders[col] = {v: i for i, v in enumerate(ordered_values)}

            for row in all_rows:
                val = row.get(col, "")
                row[col] = str(encoders[col].get(val, 0))

        temp_path = self._create_temp_dataset(headers, all_rows)
        self.file_path = temp_path

        existing_numeric = [c for c in self.metadata.get("numeric_columns", []) if c not in categorical_cols]
        existing_numeric.extend(categorical_cols)
        self.metadata["numeric_columns"] = existing_numeric
        self.metadata["input_shape"] = [len(existing_numeric)]

        if "ordinal_encoders" not in self.metadata:
            self.metadata["ordinal_encoders"] = {}
        self.metadata["ordinal_encoders"].update(encoders)

        return {
            "affected_samples": len(all_rows),
            "affected_columns": len(categorical_cols),
            "message": f"Ordinal encoded {len(categorical_cols)} columns"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_target_encode(self, params):
    columns_str = params.get("columns", "")
    label_col = params.get("label_col", "")
    smoothing = params.get("smoothing", 1.0)

    if self.dataset_type == "tabular_csv":
        if not columns_str:
            categorical_cols = [
                c for c in self.metadata.get("feature_columns", [])
                if c not in self.metadata.get("numeric_columns", [])
                and c != self.metadata.get("label_column")
            ]
        else:
            categorical_cols = [c.strip() for c in columns_str.split(",") if c.strip()]

        if not label_col:
            label_col = self.metadata.get("label_column")

        if not label_col or not categorical_cols:
            return {"affected_samples": 0, "affected_columns": 0}

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            all_rows = list(reader)

        label_values = []
        for row in all_rows:
            try:
                label_values.append(float(row.get(label_col, 0)))
            except (ValueError, TypeError):
                label_values.append(0.0)

        encoders = {}

        for col in categorical_cols:
            if col not in all_rows[0]:
                continue

            category_targets = {}
            category_counts = {}

            for row, lbl in zip(all_rows, label_values):
                cat = row.get(col, "")
                if cat not in category_targets:
                    category_targets[cat] = 0.0
                    category_counts[cat] = 0
                category_targets[cat] += lbl
                category_counts[cat] += 1

            global_mean = np.mean(label_values)
            for cat in category_targets:
                n = category_counts[cat]
                category_targets[cat] = (category_targets[cat] + smoothing * global_mean) / (n + smoothing)

            encoders[col] = category_targets

            for row, lbl in zip(all_rows, label_values):
                cat = row.get(col, "")
                row[col] = str(round(category_targets.get(cat, global_mean), 6))

        temp_path = self._create_temp_dataset(headers, all_rows)
        self.file_path = temp_path

        existing_numeric = [c for c in self.metadata.get("numeric_columns", []) if c not in categorical_cols]
        existing_numeric.extend(categorical_cols)
        self.metadata["numeric_columns"] = existing_numeric
        self.metadata["input_shape"] = [len(existing_numeric)]

        if "target_encoders" not in self.metadata:
            self.metadata["target_encoders"] = {}
        self.metadata["target_encoders"].update(encoders)

        return {
            "affected_samples": len(all_rows),
            "affected_columns": len(categorical_cols),
            "message": f"Target encoded {len(categorical_cols)} columns (smoothing={smoothing})"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_frequency_encode(self, params):
    columns_str = params.get("columns", "")

    if self.dataset_type == "tabular_csv":
        if not columns_str:
            categorical_cols = [
                c for c in self.metadata.get("feature_columns", [])
                if c not in self.metadata.get("numeric_columns", [])
                and c != self.metadata.get("label_column")
            ]
        else:
            categorical_cols = [c.strip() for c in columns_str.split(",") if c.strip()]

        if not categorical_cols:
            return {"affected_samples": 0, "affected_columns": 0}

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            all_rows = list(reader)

        encoders = {}

        for col in categorical_cols:
            if col not in all_rows[0]:
                continue

            value_counts = {}
            for row in all_rows:
                val = row.get(col, "")
                value_counts[val] = value_counts.get(val, 0) + 1

            total = len(all_rows)
            encoders[col] = {v: c / total for v, c in value_counts.items()}

            for row in all_rows:
                val = row.get(col, "")
                row[col] = str(round(encoders[col].get(val, 0), 6))

        temp_path = self._create_temp_dataset(headers, all_rows)
        self.file_path = temp_path

        existing_numeric = [c for c in self.metadata.get("numeric_columns", []) if c not in categorical_cols]
        existing_numeric.extend(categorical_cols)
        self.metadata["numeric_columns"] = existing_numeric
        self.metadata["input_shape"] = [len(existing_numeric)]

        if "frequency_encoders" not in self.metadata:
            self.metadata["frequency_encoders"] = {}
        self.metadata["frequency_encoders"].update(encoders)

        return {
            "affected_samples": len(all_rows),
            "affected_columns": len(categorical_cols),
            "message": f"Frequency encoded {len(categorical_cols)} columns"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_binary_encode(self, params):
    columns_str = params.get("columns", "")

    if self.dataset_type == "tabular_csv":
        if not columns_str:
            categorical_cols = [
                c for c in self.metadata.get("feature_columns", [])
                if c not in self.metadata.get("numeric_columns", [])
                and c != self.metadata.get("label_column")
            ]
        else:
            categorical_cols = [c.strip() for c in columns_str.split(",") if c.strip()]

        if not categorical_cols:
            return {"affected_samples": 0, "affected_columns": 0}

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            all_rows = list(reader)

        encoders = {}

        for col in categorical_cols:
            if col not in all_rows[0]:
                continue

            values = sorted(set(row.get(col, "") for row in all_rows if row.get(col, "")))

            value_to_int = {v: i for i, v in enumerate(values)}
            max_bits = bin(len(values) - 1)[2:] if len(values) > 1 else "1"
            bit_width = max(len(max_bits), 1)

            encoders[col] = {"values": values, "bit_width": bit_width}

            for row in all_rows:
                val = row.get(col, "")
                int_val = value_to_int.get(val, 0)
                binary = bin(int_val)[2:].zfill(bit_width)

                row[col] = binary[0]

                for i, bit in enumerate(binary[1:]):
                    new_col = f"{col}_bit{i+1}"
                    if new_col not in headers:
                        headers.append(new_col)
                    row[new_col] = bit

        temp_path = self._create_temp_dataset(headers, all_rows)
        self.file_path = temp_path

        new_bit_cols = []
        for col, info in encoders.items():
            if "bit_width" in info:
                for i in range(1, info["bit_width"]):
                    new_bit_cols.append(f"{col}_bit{i}")

        existing_numeric = [c for c in self.metadata.get("numeric_columns", []) if c not in categorical_cols]
        existing_numeric.extend(categorical_cols)
        existing_numeric.extend(new_bit_cols)
        self.metadata["numeric_columns"] = existing_numeric
        self.metadata["input_shape"] = [len(existing_numeric)]

        if "binary_encoders" not in self.metadata:
            self.metadata["binary_encoders"] = {}
        self.metadata["binary_encoders"].update(encoders)

        return {
            "affected_samples": len(all_rows),
            "affected_columns": len(categorical_cols),
            "message": f"Binary encoded {len(categorical_cols)} columns"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _apply_hash_encode(self, params):
    columns_str = params.get("columns", "")
    n_components = params.get("n_components", 8)
    signed = params.get("signed", False)

    if self.dataset_type == "tabular_csv":
        if not columns_str:
            categorical_cols = [
                c for c in self.metadata.get("feature_columns", [])
                if c not in self.metadata.get("numeric_columns", [])
                and c != self.metadata.get("label_column")
            ]
        else:
            categorical_cols = [c.strip() for c in columns_str.split(",") if c.strip()]

        if not categorical_cols:
            return {"affected_samples": 0, "affected_columns": 0}

        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            all_rows = list(reader)

        encoders = {}

        for col in categorical_cols:
            if col not in all_rows[0]:
                continue

            encoders[col] = {"n_components": n_components, "signed": signed}

            for row in all_rows:
                val = row.get(col, "")
                hash_val = int(hashlib.md5(val.encode()).hexdigest(), 16)

                bits = []
                for i in range(n_components):
                    bit = (hash_val >> i) & 1
                    if signed:
                        bit = bit * 2 - 1
                    bits.append(str(bit))

                row[col] = ",".join(bits)

        temp_path = self._create_temp_dataset(headers, all_rows)
        self.file_path = temp_path

        existing_numeric = [c for c in self.metadata.get("numeric_columns", []) if c not in categorical_cols]
        existing_numeric.extend(categorical_cols)
        self.metadata["numeric_columns"] = existing_numeric
        self.metadata["input_shape"] = [len(existing_numeric)]

        if "hash_encoders" not in self.metadata:
            self.metadata["hash_encoders"] = {}
        self.metadata["hash_encoders"].update(encoders)

        return {
            "affected_samples": len(all_rows),
            "affected_columns": len(categorical_cols),
            "message": f"Hash encoded {len(categorical_cols)} columns into {n_components} components"
        }

    return {"affected_samples": 0, "affected_columns": 0}


def _create_temp_dataset(self, headers: list[str], rows: list[dict[str, str]]) -> str:
    dataset_id = hashlib.md5(f"processed_{self.source_id}_{time.time()}".encode()).hexdigest()[:12]
    dest_path = os.path.join(self.ds_manager.DATASETS_DIR, f"{dataset_id}.csv")

    with open(dest_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return dest_path


def _export_dataset(self):
    new_id = hashlib.md5(f"processed_{self.source_id}_{time.time()}".encode()).hexdigest()[:12]

    self.metadata["preprocessed_from"] = self.source_id
    self.metadata["transformations"] = self.transformations_applied
    self.metadata["processed_at"] = datetime.now().isoformat()

    from preprocessing.metadata import PreprocessingMetadata

    new_info: dict[str, Any] = {
        "id": new_id,
        "name": f"{self.source_info.get('name', 'dataset')}_preprocessed",
        "dataset_type": self.dataset_type,
        "num_samples": self.num_samples,
        "num_classes": self.num_classes,
        "input_shape": self.metadata.get("input_shape", self.source_info.get("input_shape", [])),
        "class_names": self.source_info.get("class_names", []),
        "file_path": self.file_path,
        "file_size": os.path.getsize(self.file_path),
        "created_at": datetime.now().isoformat(),
        "status": "ready",
        "metadata": self.metadata
    }

    from datasets.info import DatasetInfo
    self.ds_manager._datasets[new_id] = DatasetInfo(**new_info)
    self.ds_manager._save_registry()

    return PreprocessingResult(
        success=True,
        message=f"Preprocessing completed: {len(self.transformations_applied)} transformations applied",
        affected_samples=self.affected_samples,
        affected_columns=self.affected_columns,
        new_dataset_id=new_id
    )
