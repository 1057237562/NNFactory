import importlib.util
import os
import csv
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from code_generator import Blueprint, CodeGenerator


class CustomEvaluator:
    """Evaluate trained models on data without the full training pipeline.

    Supports image classification and tabular inference. Builds models from
    blueprints, loads trained weights, and produces predictions.
    """

    TEMP_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")

    def __init__(self, blueprint: Blueprint) -> None:
        self.blueprint = blueprint
        self.code_generator = CodeGenerator(blueprint)
        self._temp_path: str | None = None
        os.makedirs(self.TEMP_DIR, exist_ok=True)

    def build_model(self, device: str = "cpu") -> Any:
        """Build a PyTorch model from the blueprint and set to eval mode.

        Args:
            device: Target device string ("cpu", "cuda", etc.).

        Returns:
            Instantiated model on the requested device.
        """
        if not self.blueprint.layers:
            raise ValueError("No layers in blueprint")

        code = self.code_generator.generate()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.blueprint.model_name}_{timestamp}.py"
        self._temp_path = os.path.join(self.TEMP_DIR, filename)
        with open(self._temp_path, "w") as f:
            f.write(code)

        module = self._load_module(self._temp_path)
        model_class = getattr(module, self.blueprint.model_name)
        model = model_class()
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_module(path: str) -> Any:
        """Import a Python file as a module at runtime."""
        spec = importlib.util.spec_from_file_location("_nnfactory_eval_model", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_weights(self, weights_filename: str) -> dict[str, Any]:
        """Load trained weights into a fresh model instance.

        Validates that the state dict keys match the model architecture.

        Args:
            weights_filename: Path to the .pth weights file.

        Returns:
            Dict with ``valid`` (bool) and optional ``errors`` list.
        """
        if not os.path.exists(weights_filename):
            return {"valid": False, "errors": [f"Weights file not found: {weights_filename}"]}

        model = self.build_model()
        try:
            state_dict = torch.load(weights_filename, map_location="cpu")
        except Exception:
            return {"valid": False, "errors": ["Failed to load weights file"]}

        model_state = model.state_dict()
        missing_keys = set(model_state.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_state.keys())

        if missing_keys or unexpected_keys:
            errors: list[str] = []
            if missing_keys:
                errors.append(f"Architecture mismatch: missing keys {sorted(missing_keys)}")
            if unexpected_keys:
                errors.append(f"Architecture mismatch: unexpected keys {sorted(unexpected_keys)}")
            return {"valid": False, "errors": errors}

        model.load_state_dict(state_dict)
        return {"valid": True}

    def detect_type(self) -> dict[str, Any]:
        """Analyze blueprint layers to classify the model type.

        Returns:
            Dict with keys:
                ``type`` — ``"image"``, ``"tabular"``, or ``"unknown"``.
                ``input_shape`` — list (image) or dict with ``in_features`` (tabular).
                ``num_classes`` — inferred from last linear layer.
        """
        if not self.blueprint.layers:
            raise ValueError("No layers in blueprint")

        layers = self.blueprint.layers
        layer_types = [l.type.lower() for l in layers]

        image_keywords = {
            "conv2d", "conv1d", "conv3d",
            "maxpool2d", "maxpool1d",
            "avgpool2d", "avgpool1d",
            "adaptive_avgpool2d", "adaptive_avgpool1d",
            "convtranspose2d", "convtranspose1d",
        }
        has_image_layer = any(t in image_keywords for t in layer_types)

        linear_layers = [l for l in layers if l.type.lower() == "linear"]
        num_classes = linear_layers[-1].params.get("out_features", 2) if linear_layers else 2

        if has_image_layer:
            input_layers = [l for l in layers if l.type.lower() == "input"]
            if input_layers:
                inp = input_layers[0]
                shape_from_params = inp.params.get("shape", None)
                if shape_from_params:
                    input_shape = shape_from_params
                else:
                    input_shape = inp.params.get("input_shape", [3, 224, 224])
            else:
                input_shape = [3, 224, 224]
            return {"type": "image", "input_shape": input_shape, "num_classes": num_classes}

        non_input_layers = [l for l in layers if l.type.lower() != "input"]
        if non_input_layers and non_input_layers[0].type.lower() == "linear":
            in_features = non_input_layers[0].params.get("in_features", 10)
            return {
                "type": "tabular",
                "input_shape": {"in_features": in_features},
                "num_classes": num_classes,
            }

        return {"type": "unknown", "input_shape": [], "num_classes": num_classes}

    def evaluate_images(
        self, image_paths: list[str], top_k: int = 5
    ) -> dict[str, Any]:
        """Run image classification inference on a list of image files.

        Args:
            image_paths: Paths to image files.
            top_k: Number of top predictions per image.

        Returns:
            Dict with ``predictions`` list and ``valid`` status.
        """
        type_info = self.detect_type()

        if not image_paths:
            return {"predictions": [], "valid": True}

        model = self.build_model()
        model.eval()

        predictions: list[list[dict[str, Any]]] = []
        with torch.inference_mode():
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")

                    if type_info["type"] == "image":
                        input_shape = type_info.get("input_shape", [3, 224, 224])
                        h = input_shape[1] if len(input_shape) >= 2 else 224
                        w = input_shape[2] if len(input_shape) >= 3 else 224
                        transform = transforms.Compose([
                            transforms.Resize((h, w)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            ),
                        ])
                        tensor = transform(img).unsqueeze(0)
                    else:
                        tensor = transforms.ToTensor()(img).unsqueeze(0)
                        tensor = tensor.view(tensor.size(0), -1)

                    output = model(tensor)
                    probs = F.softmax(output, dim=1)
                    top_probs, top_indices = torch.topk(
                        probs, min(top_k, probs.size(1)), dim=1
                    )

                    img_preds = []
                    for i in range(top_probs.size(1)):
                        cls_idx = int(top_indices[0, i].item())
                        img_preds.append({
                            "class": cls_idx,
                            "confidence": float(top_probs[0, i].item()),
                            "label": f"Class {cls_idx}",
                        })
                    predictions.append(img_preds)
                except Exception as e:
                    predictions.append([
                        {"class": -1, "confidence": 0.0, "label": f"Error: {e}"},
                    ])

        return {"predictions": predictions, "valid": True}

    def evaluate_tabular_csv(self, csv_path: str) -> dict[str, Any]:
        """Run inference on tabular CSV data and write results with prediction column.

        Detects numeric columns, applies z-score normalisation to the uploaded
        data, runs the model, and writes an output CSV with an added
        ``prediction`` column.

        Args:
            csv_path: Path to the input CSV file.

        Returns:
            Dict with ``output_path`` and ``valid`` status.
        """
        if not os.path.exists(csv_path):
            return {"valid": False, "errors": [f"CSV file not found: {csv_path}"]}

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if not rows:
            return {"valid": False, "errors": ["CSV file is empty"]}

        skip_cols = {"label", "target", "class", "prediction"}
        numeric_cols: list[str] = []
        for col in fieldnames:
            if col.lower() in skip_cols:
                continue
            try:
                float(rows[0].get(col, ""))
                numeric_cols.append(col)
            except (ValueError, TypeError):
                pass

        if not numeric_cols:
            return {"valid": False, "errors": ["No numeric columns found in CSV"]}

        n_rows = len(rows)
        n_cols = len(numeric_cols)
        x_matrix = np.zeros((n_rows, n_cols))
        for i, row in enumerate(rows):
            for j, col in enumerate(numeric_cols):
                try:
                    x_matrix[i, j] = float(row.get(col, 0))
                except (ValueError, TypeError):
                    x_matrix[i, j] = 0.0

        mean = x_matrix.mean(axis=0)
        std = x_matrix.std(axis=0) + 1e-8
        x_normalized = (x_matrix - mean) / std

        x_tensor = torch.tensor(x_normalized, dtype=torch.float32)

        model = self.build_model()
        model.eval()

        with torch.inference_mode():
            output = model(x_tensor)
            if output.dim() > 1:
                pred_indices = output.argmax(dim=1).cpu().numpy()
            else:
                pred_indices = (output > 0).long().cpu().numpy()

        output_dir = os.path.dirname(os.path.abspath(csv_path))
        output_filename = f"eval_{os.path.basename(csv_path)}"
        output_path = os.path.join(output_dir, output_filename)

        new_fieldnames = list(fieldnames) + ["prediction"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            for i, row in enumerate(rows):
                out_row = dict(row)
                out_row["prediction"] = int(pred_indices[i])
                writer.writerow(out_row)

        return {"output_path": output_path, "valid": True}

    def evaluate_tabular_single(self, values: list[float]) -> dict[str, Any]:
        """Run inference on a single row of tabular data.

        Args:
            values: Feature values matching the model's ``in_features``.

        Returns:
            Dict with ``predictions`` list and ``valid`` status.
        """
        model = self.build_model()
        model.eval()

        x_tensor = torch.tensor([values], dtype=torch.float32)

        with torch.inference_mode():
            output = model(x_tensor)
            probs = F.softmax(output, dim=1)
            k = min(probs.size(1), 5)
            top_probs, top_indices = torch.topk(probs, k, dim=1)

        preds = []
        for i in range(k):
            cls_idx = int(top_indices[0, i].item())
            preds.append({
                "class": cls_idx,
                "confidence": float(top_probs[0, i].item()),
                "label": f"Class {cls_idx}",
            })

        return {"predictions": preds, "valid": True}
