import os
import json
import shutil
import hashlib
import time
import numpy as np
from PIL import Image
from typing import Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class DatasetInfo:
    id: str
    name: str
    dataset_type: str
    num_samples: int
    num_classes: int = 0
    input_shape: list[int] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    file_path: str = ""
    file_size: int = 0
    created_at: str = ""
    status: str = "ready"
    split_info: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetManager:
    DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", ".dataset_cache")
    MAX_ROWS_FOR_STATS = 50000

    def __init__(self):
        os.makedirs(self.DATASETS_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._datasets: dict[str, DatasetInfo] = {}
        self._load_registry()
        self._viz_cache: dict[str, tuple[Any, float]] = {}
        self._col_cache: dict[str, tuple[Any, float]] = {}
        self._cache_ttl = 300

    def _registry_path(self) -> str:
        return os.path.join(self.CACHE_DIR, "registry.json")

    def _load_registry(self):
        path = self._registry_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for item in data:
                    info = DatasetInfo(**item)
                    self._datasets[info.id] = info
            except Exception:
                pass

    def _save_registry(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        with open(self._registry_path(), "w") as f:
            json.dump([d.to_dict() for d in self._datasets.values()], f, indent=2)

    def _generate_id(self, name: str) -> str:
        ts = datetime.now().isoformat()
        return hashlib.md5(f"{name}_{ts}".encode()).hexdigest()[:12]

    def _format_size(self, size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def list_datasets(self) -> list[dict]:
        return [d.to_dict() for d in self._datasets.values()]

    def get_dataset(self, dataset_id: str) -> dict | None:
        info = self._datasets.get(dataset_id)
        return info.to_dict() if info else None

    def delete_dataset(self, dataset_id: str) -> dict:
        info = self._datasets.get(dataset_id)
        if not info:
            return {"valid": False, "errors": ["Dataset not found"]}

        if os.path.exists(info.file_path):
            if os.path.isdir(info.file_path):
                shutil.rmtree(info.file_path)
            else:
                os.remove(info.file_path)

        cache_dir = os.path.join(self.CACHE_DIR, dataset_id)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        del self._datasets[dataset_id]
        self._save_registry()
        return {"valid": True, "message": f"Dataset '{info.name}' deleted"}

    def purge_all(self) -> dict:
        count = len(self._datasets)
        for ds_id in list(self._datasets.keys()):
            self.delete_dataset(ds_id)
        if os.path.exists(self.CACHE_DIR):
            shutil.rmtree(self.CACHE_DIR)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        return {"valid": True, "message": f"All {count} datasets purged"}

    def load_from_folder(self, folder_path: str, name: str = None) -> dict:
        if not os.path.isdir(folder_path):
            return {"valid": False, "errors": ["Path is not a directory"]}

        if name is None:
            name = os.path.basename(folder_path.rstrip("/\\"))

        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        if subdirs:
            return self._load_image_folder(folder_path, subdirs, name)
        else:
            return self._load_flat_folder(folder_path, name)

    def _load_image_folder(self, folder_path: str, subdirs: list[str], name: str) -> dict:
        class_names = sorted(subdirs)
        num_classes = len(class_names)
        total_samples = 0
        split_info = {}
        input_shapes = []

        for cls_name in class_names:
            cls_path = os.path.join(folder_path, cls_name)
            files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"))]
            split_info[cls_name] = len(files)
            total_samples += len(files)

            for f in files[:5]:
                try:
                    img = Image.open(os.path.join(cls_path, f))
                    arr = np.array(img)
                    if arr.ndim == 3:
                        shape = [arr.shape[2], arr.shape[0], arr.shape[1]]
                    else:
                        shape = [1, arr.shape[0], arr.shape[1]]
                    input_shapes.append(shape)
                except Exception:
                    pass

        if input_shapes:
            common_shape = max(set(map(tuple, input_shapes)), key=input_shapes.count)
            input_shape = list(common_shape)
        else:
            input_shape = [3, 224, 224]

        dataset_id = self._generate_id(name)
        dest_dir = os.path.join(self.DATASETS_DIR, dataset_id)
        shutil.copytree(folder_path, dest_dir)

        total_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(dest_dir)
            for f in filenames
        )

        info = DatasetInfo(
            id=dataset_id,
            name=name,
            dataset_type="image_classification",
            num_samples=total_samples,
            num_classes=num_classes,
            input_shape=input_shape,
            class_names=class_names,
            file_path=dest_dir,
            file_size=total_size,
            created_at=datetime.now().isoformat(),
            status="ready",
            split_info=split_info,
            metadata={"folder_structure": "class_subfolders"}
        )

        self._datasets[dataset_id] = info
        self._save_registry()
        return {"valid": True, "dataset": info.to_dict()}

    def _load_flat_folder(self, folder_path: str, name: str) -> dict:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"))]
        total_samples = len(files)

        input_shapes = []
        for f in files[:10]:
            try:
                img = Image.open(os.path.join(folder_path, f))
                arr = np.array(img)
                if arr.ndim == 3:
                    shape = [arr.shape[2], arr.shape[0], arr.shape[1]]
                else:
                    shape = [1, arr.shape[0], arr.shape[1]]
                input_shapes.append(shape)
            except Exception:
                pass

        if input_shapes:
            common_shape = max(set(map(tuple, input_shapes)), key=input_shapes.count)
            input_shape = list(common_shape)
        else:
            input_shape = [3, 224, 224]

        dataset_id = self._generate_id(name)
        dest_dir = os.path.join(self.DATASETS_DIR, dataset_id)
        shutil.copytree(folder_path, dest_dir)

        total_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(dest_dir)
            for f in filenames
        )

        info = DatasetInfo(
            id=dataset_id,
            name=name,
            dataset_type="image_folder",
            num_samples=total_samples,
            num_classes=0,
            input_shape=input_shape,
            class_names=[],
            file_path=dest_dir,
            file_size=total_size,
            created_at=datetime.now().isoformat(),
            status="ready",
            split_info={"total": total_samples},
            metadata={"folder_structure": "flat"}
        )

        self._datasets[dataset_id] = info
        self._save_registry()
        return {"valid": True, "dataset": info.to_dict()}

    def load_from_csv(self, file_path: str, name: str = None, label_column: str = None) -> dict:
        import csv

        if not os.path.isfile(file_path):
            return {"valid": False, "errors": ["File not found"]}

        if name is None:
            name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                if not headers:
                    return {"valid": False, "errors": ["CSV has no headers row"]}

                rows = list(reader)

            if not rows:
                return {"valid": False, "errors": ["CSV has no data rows"]}

            num_samples = len(rows)
            num_features = len(headers)

            if label_column and label_column in headers:
                feature_cols = [h for h in headers if h != label_column]
                labels = [row[label_column] for row in rows]
                unique_labels = sorted(set(labels))
                num_classes = len(unique_labels)
                class_names = [str(l) for l in unique_labels]
            else:
                feature_cols = headers
                num_classes = 0
                class_names = []
                labels = []

            numeric_cols = []
            for col in feature_cols:
                try:
                    float(rows[0][col])
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    pass

            input_shape = [len(numeric_cols)] if numeric_cols else [num_features]

            dataset_id = self._generate_id(name)
            dest_path = os.path.join(self.DATASETS_DIR, f"{dataset_id}.csv")
            shutil.copy2(file_path, dest_path)

            file_size = os.path.getsize(dest_path)

            label_dist = {}
            if labels:
                for lbl in labels:
                    label_dist[str(lbl)] = label_dist.get(str(lbl), 0) + 1

            info = DatasetInfo(
                id=dataset_id,
                name=name,
                dataset_type="tabular_csv",
                num_samples=num_samples,
                num_classes=num_classes,
                input_shape=input_shape,
                class_names=class_names,
                file_path=dest_path,
                file_size=file_size,
                created_at=datetime.now().isoformat(),
                status="ready",
                split_info=label_dist,
                metadata={
                    "headers": headers,
                    "feature_columns": feature_cols,
                    "label_column": label_column,
                    "numeric_columns": numeric_cols,
                    "label_distribution": label_dist
                }
            )

            self._datasets[dataset_id] = info
            self._save_registry()
            return {"valid": True, "dataset": info.to_dict()}

        except Exception as e:
            return {"valid": False, "errors": [f"Failed to parse CSV: {str(e)}"]}

    def get_preview(self, dataset_id: str, limit: int = 10) -> dict:
        info = self._datasets.get(dataset_id)
        if not info:
            return {"valid": False, "errors": ["Dataset not found"]}

        if info.dataset_type == "tabular_csv":
            return self._preview_csv(info, limit)
        elif info.dataset_type in ("image_classification", "image_folder"):
            return self._preview_images(info, limit)
        else:
            return {"valid": False, "errors": ["Unsupported dataset type for preview"]}

    def _preview_csv(self, info: DatasetInfo, limit: int) -> dict:
        import csv
        try:
            with open(info.file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)[:limit]
            return {
                "valid": True,
                "headers": info.metadata.get("headers", []),
                "rows": rows,
                "total": info.num_samples,
                "showing": len(rows)
            }
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}

    def _preview_images(self, info: DatasetInfo, limit: int) -> dict:
        import base64
        from io import BytesIO

        images = []
        if info.dataset_type == "image_classification":
            for cls_name in info.class_names:
                cls_path = os.path.join(info.file_path, cls_name)
                if not os.path.isdir(cls_path):
                    continue
                files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
                for f in files[:max(1, limit // max(len(info.class_names), 1))]:
                    img_path = os.path.join(cls_path, f)
                    images.append({
                        "path": img_path,
                        "filename": f,
                        "class": cls_name,
                        "size": os.path.getsize(img_path)
                    })
        else:
            files = [f for f in os.listdir(info.file_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
            for f in files[:limit]:
                img_path = os.path.join(info.file_path, f)
                images.append({
                    "path": img_path,
                    "filename": f,
                    "class": None,
                    "size": os.path.getsize(img_path)
                })

        for img in images[:10]:
            try:
                with Image.open(img["path"]) as pil_img:
                    pil_img.thumbnail((128, 128))
                    buf = BytesIO()
                    fmt = "JPEG" if pil_img.mode == "RGB" else "PNG"
                    pil_img.save(buf, format=fmt)
                    img["thumbnail"] = base64.b64encode(buf.getvalue()).decode("utf-8")
                    img["width"], img["height"] = pil_img.size
            except Exception:
                img["thumbnail"] = None

        return {
            "valid": True,
            "images": images,
            "total_images": info.num_samples,
            "showing": len(images)
        }

    def get_visualization(self, dataset_id: str) -> dict:
        now = time.time()
        cache_key = f"viz_{dataset_id}"
        if cache_key in self._viz_cache:
            cached, ts = self._viz_cache[cache_key]
            if now - ts < self._cache_ttl:
                return {"valid": True, "visualization": cached}

        info = self._datasets.get(dataset_id)
        if not info:
            return {"valid": False, "errors": ["Dataset not found"]}

        viz = {
            "id": info.id,
            "name": info.name,
            "type": info.dataset_type,
            "num_samples": info.num_samples,
            "num_classes": info.num_classes,
            "input_shape": info.input_shape,
        }

        if info.dataset_type == "tabular_csv":
            viz["feature_count"] = len(info.metadata.get("feature_columns", []))
            viz["label_column"] = info.metadata.get("label_column")
            viz["numeric_columns"] = info.metadata.get("numeric_columns", [])
            viz["categorical_columns"] = [
                c for c in info.metadata.get("feature_columns", [])
                if c not in info.metadata.get("numeric_columns", [])
            ]
            viz["label_distribution"] = info.metadata.get("label_distribution", {})

            try:
                import csv
                with open(info.file_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                numeric_cols = info.metadata.get("numeric_columns", [])
                stats = {}
                for col in numeric_cols[:20]:
                    values = [float(r[col]) for r in rows if r.get(col, "") != ""]
                    if values:
                        arr = np.array(values)
                        stats[col] = {
                            "min": float(np.min(arr)),
                            "max": float(np.max(arr)),
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr)),
                            "median": float(np.median(arr)),
                        }
                viz["column_statistics"] = stats
            except Exception:
                pass

        elif info.dataset_type == "image_classification":
            class_dist = {}
            for cls_name in info.class_names:
                cls_path = os.path.join(info.file_path, cls_name)
                if os.path.isdir(cls_path):
                    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
                    class_dist[cls_name] = len(files)
            viz["class_distribution"] = class_dist

            shapes = {}
            for cls_name in info.class_names[:5]:
                cls_path = os.path.join(info.file_path, cls_name)
                if not os.path.isdir(cls_path):
                    continue
                files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))][:5]
                for f in files:
                    try:
                        img = Image.open(os.path.join(cls_path, f))
                        arr = np.array(img)
                        key = f"{arr.shape[1]}x{arr.shape[0]}"
                        shapes[key] = shapes.get(key, 0) + 1
                    except Exception:
                        pass
            viz["shape_distribution"] = shapes

        elif info.dataset_type == "image_folder":
            files = [f for f in os.listdir(info.file_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
            shapes = {}
            for f in files[:20]:
                try:
                    img = Image.open(os.path.join(info.file_path, f))
                    arr = np.array(img)
                    key = f"{arr.shape[1]}x{arr.shape[0]}"
                    shapes[key] = shapes.get(key, 0) + 1
                except Exception:
                    pass
            viz["shape_distribution"] = shapes

        self._viz_cache[cache_key] = (viz, now)
        return {"valid": True, "visualization": viz}

    def _load_csv_rows(self, info, max_rows: int = None):
        import csv
        with open(info.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if max_rows and max_rows > 0:
                rows = []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(row)
                return rows
            return list(reader)

    def get_column_stats(self, dataset_id: str, column: str = None) -> dict:
        now = time.time()
        cache_key = f"col_{dataset_id}_{column or 'all'}"
        if cache_key in self._col_cache:
            cached, ts = self._col_cache[cache_key]
            if now - ts < self._cache_ttl:
                return {"valid": True, "column_stats": cached}

        info = self._datasets.get(dataset_id)
        if not info:
            return {"valid": False, "errors": ["Dataset not found"]}

        if info.dataset_type != "tabular_csv":
            return {"valid": False, "errors": ["Only tabular CSV datasets support column statistics"]}

        import traceback
        try:
            rows = self._load_csv_rows(info, self.MAX_ROWS_FOR_STATS)

            if not rows:
                return {"valid": False, "errors": ["No data rows"]}

            raw_headers = list(rows[0].keys())
            header_map = {}
            for h in raw_headers:
                clean = h.strip().replace('\r', '').replace('\n', '')
                header_map[clean] = h

            label_col = info.metadata.get("label_column")
            if label_col:
                lc = label_col.strip()
                if lc in header_map:
                    label_col = header_map[lc]

            def resolve_col(name):
                clean = name.strip().replace('\r', '').replace('\n', '')
                return header_map.get(clean)

            def get_val(row, col_name):
                if row is None:
                    return ""
                v = row.get(col_name)
                return v.strip() if v else ""

            def safe_float_values(rows_list, col_name):
                vals = []
                for r in rows_list:
                    raw = get_val(r, col_name)
                    if raw == "":
                        continue
                    try:
                        vals.append(float(raw))
                    except (ValueError, TypeError):
                        pass
                return vals

            numeric_cols = []
            categorical_cols = []
            for actual_h in raw_headers:
                if label_col and actual_h == label_col:
                    categorical_cols.append(actual_h)
                    continue
                numeric_count = 0
                total_count = 0
                for r in rows:
                    val = get_val(r, actual_h)
                    if val == "":
                        continue
                    total_count += 1
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                if total_count > 0 and numeric_count / total_count > 0.8:
                    numeric_cols.append(actual_h)
                else:
                    categorical_cols.append(actual_h)

            clean_headers = [h.strip() for h in raw_headers]

            result = {
                "numeric_columns": [h.strip() for h in numeric_cols],
                "categorical_columns": [h.strip() for h in categorical_cols],
                "label_column": label_col.strip() if label_col else None,
                "all_columns": clean_headers,
            }

            corr_matrix = {}
            if len(numeric_cols) > 1:
                for col in numeric_cols:
                    corr_matrix[col.strip()] = {}
                for i, col_a in enumerate(numeric_cols):
                    vals_a = np.array(safe_float_values(rows, col_a), dtype=np.float64)
                    for col_b in numeric_cols[i:]:
                        vals_b = np.array(safe_float_values(rows, col_b), dtype=np.float64)
                        min_len = min(len(vals_a), len(vals_b))
                        if min_len < 2:
                            corr = 0.0
                        else:
                            corr = float(np.corrcoef(vals_a[:min_len], vals_b[:min_len])[0, 1])
                            if np.isnan(corr):
                                corr = 0.0
                        corr_matrix[col_a.strip()][col_b.strip()] = round(corr, 4)
                        if col_a != col_b:
                            corr_matrix[col_b.strip()][col_a.strip()] = round(corr, 4)
            elif len(numeric_cols) == 1:
                corr_matrix[numeric_cols[0].strip()] = {numeric_cols[0].strip(): 1.0}

            result["correlation_matrix"] = corr_matrix

            if column:
                actual_col = resolve_col(column)
                if actual_col is None:
                    return {"valid": False, "errors": [f"Column '{column}' not found. Available: {', '.join(clean_headers)}"]}

                if actual_col in numeric_cols:
                    values = np.array(safe_float_values(rows, actual_col), dtype=np.float64)
                    if len(values) > 0:
                        hist, bin_edges = np.histogram(values, bins="auto")
                        result["histogram"] = {
                            "counts": hist.tolist(),
                            "bin_edges": bin_edges.tolist(),
                            "type": "numeric",
                        }
                        result["statistics"] = {
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "median": float(np.median(values)),
                            "q25": float(np.percentile(values, 25)),
                            "q75": float(np.percentile(values, 75)),
                        }
                    result["relations"] = self._compute_column_relations(rows, actual_col, numeric_cols, categorical_cols, label_col, resolve_col, get_val, safe_float_values)
                elif actual_col in categorical_cols or actual_col == label_col:
                    value_counts = {}
                    for r in rows:
                        val = get_val(r, actual_col)
                        if val != "":
                            value_counts[val] = value_counts.get(val, 0) + 1
                    sorted_counts = dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True))
                    result["value_counts"] = sorted_counts
                    result["histogram"] = {
                        "type": "categorical",
                        "categories": list(sorted_counts.keys()),
                        "counts": list(sorted_counts.values()),
                    }
                    result["relations"] = self._compute_column_relations(rows, actual_col, numeric_cols, categorical_cols, label_col, resolve_col, get_val, safe_float_values)

            self._col_cache[cache_key] = (result, now)
            return {"valid": True, "column_stats": result}

        except KeyError as e:
            return {"valid": False, "errors": [f"KeyError: {e}\n{traceback.format_exc()}"]}
        except Exception as e:
            return {"valid": False, "errors": [f"Error: {e}\n{traceback.format_exc()}"]}

    def _compute_column_relations(self, rows, source_col, numeric_cols, categorical_cols, label_col, resolve_col, get_val, safe_float_values):
        relations = {"numeric": {}, "categorical": {}}
        max_categories = 50

        for num_col in numeric_cols:
            if num_col == source_col:
                continue
            grouped = {}
            for r in rows:
                cat_val = get_val(r, source_col)
                if cat_val == "":
                    continue
                num_raw = get_val(r, num_col)
                if num_raw == "":
                    continue
                try:
                    num_val = float(num_raw)
                except (ValueError, TypeError):
                    continue
                if cat_val not in grouped:
                    grouped[cat_val] = []
                grouped[cat_val].append(num_val)

            summary = {}
            for cat_val, vals in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:max_categories]:
                arr = np.array(vals, dtype=np.float64)
                if len(arr) > 0:
                    summary[cat_val] = {
                        "count": len(arr),
                        "mean": round(float(np.mean(arr)), 4),
                        "std": round(float(np.std(arr)), 4),
                        "min": round(float(np.min(arr)), 4),
                        "max": round(float(np.max(arr)), 4),
                    }
            if summary:
                relations["numeric"][num_col] = summary

        for cat_col in categorical_cols:
            if cat_col == source_col:
                continue
            contingency = {}
            for r in rows:
                src_val = get_val(r, source_col)
                other_val = get_val(r, cat_col)
                if src_val == "" or other_val == "":
                    continue
                if src_val not in contingency:
                    contingency[src_val] = {}
                contingency[src_val][other_val] = contingency[src_val].get(other_val, 0) + 1

            top_src = sorted(contingency.keys(), key=lambda k: sum(contingency[k].values()), reverse=True)[:max_categories]
            trimmed = {}
            for src_val in top_src:
                top_other = sorted(contingency[src_val].keys(), key=lambda k: contingency[src_val][k], reverse=True)[:max_categories]
                trimmed[src_val] = {k: contingency[src_val][k] for k in top_other}
            if trimmed:
                relations["categorical"][cat_col] = trimmed

        if label_col and label_col != source_col and label_col in categorical_cols:
            contingency = {}
            for r in rows:
                src_val = get_val(r, source_col)
                lbl_val = get_val(r, label_col)
                if src_val == "" or lbl_val == "":
                    continue
                if src_val not in contingency:
                    contingency[src_val] = {}
                contingency[src_val][lbl_val] = contingency[src_val].get(lbl_val, 0) + 1

            top_src = sorted(contingency.keys(), key=lambda k: sum(contingency[k].values()), reverse=True)[:max_categories]
            trimmed = {}
            for src_val in top_src:
                trimmed[src_val] = contingency[src_val]
            if trimmed:
                relations["categorical"]["__label__"] = trimmed

        return relations

    def get_dataloader_config(self, dataset_id: str) -> dict:
        info = self._datasets.get(dataset_id)
        if not info:
            return {"valid": False, "errors": ["Dataset not found"]}

        config = {
            "dataset_id": info.id,
            "name": info.name,
            "type": info.dataset_type,
            "input_shape": info.input_shape,
            "num_classes": info.num_classes,
            "num_samples": info.num_samples,
            "class_names": info.class_names,
        }
        return {"valid": True, "config": config}
