import csv
import time
import numpy as np
from typing import Any, Optional


def get_column_stats(self, dataset_id: str, column: Optional[str] = None) -> dict[str, Any]:
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


def get_dataloader_config(self, dataset_id: str) -> dict[str, Any]:
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


def get_available_target_columns(self, dataset_id: str) -> dict[str, Any]:
    info = self._datasets.get(dataset_id)
    if not info:
        return {"valid": False, "errors": ["Dataset not found"]}

    if info.dataset_type != "tabular_csv":
        return {"valid": False, "errors": ["Only tabular CSV datasets support target column selection"]}

    import csv
    try:
        with open(info.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not headers:
                return {"valid": False, "errors": ["CSV has no headers"]}

            rows = list(reader)

        columns = []
        for col in headers:
            clean = col.strip()
            values = [row.get(col, "").strip() for row in rows[:200] if row.get(col, "").strip() != ""]
            unique_count = len(set(values))
            total_count = len(values)

            try:
                float(values[0])
                col_type = "numeric"
            except (ValueError, TypeError, IndexError):
                col_type = "categorical"

            is_current_label = (info.metadata.get("label_column") == col)
            cardinality = unique_count / total_count if total_count > 0 else 0

            columns.append({
                "name": clean,
                "type": col_type,
                "unique_count": unique_count,
                "total_count": total_count,
                "cardinality_ratio": round(cardinality, 4),
                "is_current_label": is_current_label,
                "suitable_as_target": col_type == "categorical" or (cardinality < 0.5 and unique_count <= 50),
            })

        return {"valid": True, "columns": columns, "current_label": info.metadata.get("label_column")}

    except Exception as e:
        return {"valid": False, "errors": [f"Failed to analyze columns: {str(e)}"]}


def set_label_column(self, dataset_id: str, label_column: str) -> dict[str, Any]:
    info = self._datasets.get(dataset_id)
    if not info:
        return {"valid": False, "errors": ["Dataset not found"]}

    if info.dataset_type != "tabular_csv":
        return {"valid": False, "errors": ["Only tabular CSV datasets support target column selection"]}

    import csv
    try:
        with open(info.file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not headers:
                return {"valid": False, "errors": ["CSV has no headers"]}

            rows = list(reader)

        if label_column not in headers:
            clean_headers = [h.strip() for h in headers]
            if label_column.strip() not in clean_headers:
                return {"valid": False, "errors": [f"Column '{label_column}' not found. Available: {', '.join(h.strip() for h in headers)}"]}

        labels = [row[label_column] for row in rows]
        unique_labels = sorted(set(labels))
        num_classes = len(unique_labels)
        class_names = [str(l) for l in unique_labels]

        label_dist = {}
        for lbl in labels:
            label_dist[str(lbl)] = label_dist.get(str(lbl), 0) + 1

        feature_cols = [h for h in headers if h != label_column]
        numeric_cols = []
        for col in feature_cols:
            try:
                float(rows[0][col])
                numeric_cols.append(col)
            except (ValueError, TypeError):
                pass

        info.metadata["label_column"] = label_column
        info.metadata["feature_columns"] = feature_cols
        info.metadata["numeric_columns"] = numeric_cols
        info.metadata["label_distribution"] = label_dist
        info.num_classes = num_classes
        info.class_names = class_names
        info.input_shape = [len(numeric_cols)] if numeric_cols else [len(feature_cols)]

        self._save_registry()

        cache_keys_to_clear = [f"col_{dataset_id}_{column or 'all'}" for column in [None, label_column] + feature_cols[:5]]
        for key in cache_keys_to_clear:
            self._col_cache.pop(key, None)
        self._viz_cache.pop(f"viz_{dataset_id}", None)

        return {
            "valid": True,
            "message": f"Target column set to '{label_column}'",
            "num_classes": num_classes,
            "class_names": class_names,
            "label_distribution": label_dist,
            "input_shape": info.input_shape,
        }

    except Exception as e:
        return {"valid": False, "errors": [f"Failed to set target column: {str(e)}"]}
