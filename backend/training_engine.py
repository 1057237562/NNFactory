import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import importlib.util
import os
import time
import traceback
from datetime import datetime


class TrainingEngine:
    TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")

    def __init__(self, blueprint, code_generator):
        self.blueprint = blueprint
        self.code_generator = code_generator
        self.is_training = False
        self.should_stop = False
        self._temp_path = None
        os.makedirs(self.TEMP_DIR, exist_ok=True)

    def _build_model(self):
        code = self.code_generator.generate()
        self._temp_path = self._write_temp_module(code)
        module = self._load_module(self._temp_path)
        model_class = getattr(module, self.blueprint.model_name)
        return model_class()

    def _write_temp_module(self, code):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.blueprint.model_name}_{timestamp}.py"
        path = os.path.join(self.TEMP_DIR, filename)
        with open(path, "w") as f:
            f.write(code)
        return path

    @staticmethod
    def _load_module(path):
        spec = importlib.util.spec_from_file_location("_nnfactory_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _cleanup(self):
        self._temp_path = None

    def _create_synthetic_dataset(self, config):
        input_size = config.get("input_size", [3, 224, 224])
        num_classes = config.get("num_classes", 10)
        num_samples = config.get("num_samples", 1000)
        val_ratio = config.get("val_ratio", 0.2)
        batch_size = config.get("batch_size", 32)

        train_size = int(num_samples * (1 - val_ratio))
        val_size = num_samples - train_size
        c, h, w = input_size[0], input_size[1] if len(input_size) > 1 else 1, input_size[2] if len(input_size) > 2 else 1

        x_train = torch.randn(train_size, c, h, w)
        y_train = torch.randint(0, num_classes, (train_size,))
        x_val = torch.randn(val_size, c, h, w)
        y_val = torch.randint(0, num_classes, (val_size,))

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, num_classes

    def train(self, config):
        self.is_training = True
        self.should_stop = False

        try:
            model = self._build_model()
            train_loader, val_loader, num_classes = self._create_synthetic_dataset(config)
            criterion = self._get_criterion(config.get("loss_function", "cross_entropy"), num_classes)
            optimizer = self._get_optimizer(config.get("optimizer", "adam"), model, config.get("learning_rate", 0.001), config.get("weight_decay", 0.0))
            scheduler = self._get_scheduler(config.get("scheduler", "none"), optimizer, config.get("step_size", 30), config.get("gamma", 0.1))

            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
            epochs = config.get("epochs", 10)
            total_steps = epochs * len(train_loader)
            step_count = 0
            start_time = time.time()

            for epoch in range(epochs):
                if self.should_stop:
                    break

                epoch_loss, epoch_acc, step_count, progress_events = self._train_one_epoch(
                    model, train_loader, criterion, optimizer, epoch, epochs, total_steps, step_count, start_time
                )

                for evt in progress_events:
                    yield evt

                if scheduler and config.get("scheduler", "none") != "none":
                    scheduler.step()

                val_loss, val_acc = self._evaluate_model(model, val_loader, criterion)

                history["train_loss"].append(epoch_loss)
                history["val_loss"].append(val_loss)
                history["train_acc"].append(epoch_acc)
                history["val_acc"].append(val_acc)
                history["lr"].append(optimizer.param_groups[0]["lr"])

                yield {
                    "type": "epoch_end",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "train_acc": epoch_acc,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed": time.time() - start_time,
                    "history": history
                }

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
                "history": history
            }

        except Exception as e:
            yield {"type": "error", "message": f"Training failed: {str(e)}", "traceback": traceback.format_exc()}
        finally:
            self.is_training = False
            self._cleanup()

    def _train_one_epoch(self, model, train_loader, criterion, optimizer, epoch, total_epochs, total_steps, step_count, start_time):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        progress_events = []

        for batch_x, batch_y in train_loader:
            if self.should_stop:
                break

            optimizer.zero_grad()
            output = self._forward(model, batch_x)
            loss = self._compute_loss(output, batch_y, criterion)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_total += batch_y.size(0)
            if output.dim() > 1:
                train_correct += output.max(1)[1].eq(batch_y).sum().item()
            else:
                train_correct += 1

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

    def evaluate(self, config):
        try:
            model = self._build_model()
            _, val_loader, num_classes = self._create_synthetic_dataset(config)
            criterion = self._get_criterion(config.get("loss_function", "cross_entropy"), num_classes)
            val_loss, val_acc = self._evaluate_model(model, val_loader, criterion)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            per_class = self._compute_per_class_stats(model, val_loader, num_classes)

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

    def _evaluate_model(self, model, data_loader, criterion):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                output = self._forward(model, batch_x)
                loss = self._compute_loss(output, batch_y, criterion)
                total_loss += loss.item() * batch_x.size(0)
                total += batch_y.size(0)
                if output.dim() > 1:
                    correct += output.max(1)[1].eq(batch_y).sum().item()
                else:
                    correct += 1

        return total_loss / max(total, 1), correct / max(total, 1) * 100

    def _compute_per_class_stats(self, model, data_loader, num_classes):
        model.eval()
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                output = self._forward(model, batch_x)
                if output.dim() > 1:
                    predicted = output.max(1)[1]
                    for i in range(batch_y.size(0)):
                        label = batch_y[i].item()
                        if label < num_classes:
                            class_total[label] += 1
                            if predicted[i].item() == label:
                                class_correct[label] += 1

        return [
            {"class": i, "accuracy": (class_correct[i] / max(class_total[i], 1)) * 100, "samples": class_total[i]}
            for i in range(min(num_classes, 10))
        ]

    def stop_training(self):
        self.should_stop = True

    @staticmethod
    def _forward(model, x):
        output = model(x)
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _compute_loss(output, target, criterion):
        if output.dim() == 1:
            output = output.unsqueeze(0)
        if output.shape != target.shape and output.dim() == target.dim() + 1:
            return criterion(output, target)
        if output.shape == target.shape:
            return criterion(output, target.float())
        output = output.view(output.size(0), -1)
        if output.size(1) == target.size(0):
            output = output.t()
        return criterion(output, target)

    @staticmethod
    def _get_criterion(name, num_classes):
        return {"cross_entropy": nn.CrossEntropyLoss(), "mse": nn.MSELoss(), "bce": nn.BCEWithLogitsLoss(), "l1": nn.L1Loss(), "nll": nn.NLLLoss()}.get(name, nn.CrossEntropyLoss())

    @staticmethod
    def _get_optimizer(name, model, lr, weight_decay):
        return {"adam": lambda: optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay), "adamw": lambda: optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay), "sgd": lambda: optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay), "rmsprop": lambda: optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay), "adagrad": lambda: optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)}.get(name, lambda: optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))()

    @staticmethod
    def _get_scheduler(name, optimizer, step_size, gamma):
        return {"step_lr": optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma), "cosine": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size), "exponential": optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma), "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=5)}.get(name, None)
