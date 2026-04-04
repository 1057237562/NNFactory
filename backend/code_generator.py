from typing import Any
from pydantic import BaseModel

class LayerConfig(BaseModel):
    id: str
    type: str
    params: dict[str, Any]
    position: dict[str, float]

class Connection(BaseModel):
    from_id: str
    to_id: str

class Blueprint(BaseModel):
    layers: list[LayerConfig]
    connections: list[Connection]
    model_name: str = "NeuralNetwork"
    use_jit: bool = False
    use_compile: bool = False
    device: str = "cpu"

class CodeGenerator:
    def __init__(self, blueprint: Blueprint):
        self.blueprint = blueprint
        self.layer_counter = 0

    def validate(self):
        if not self.blueprint.layers:
            raise ValueError("No layers in blueprint")
        
        layer_ids = {layer.id for layer in self.blueprint.layers}
        for conn in self.blueprint.connections:
            if conn.from_id not in layer_ids:
                raise ValueError(f"Connection references non-existent layer: {conn.from_id}")
            if conn.to_id not in layer_ids:
                raise ValueError(f"Connection references non-existent layer: {conn.to_id}")

    def _get_layer_type(self, layer_type: str) -> str:
        type_map = {
            "conv2d": "nn.Conv2d",
            "conv1d": "nn.Conv1d",
            "conv3d": "nn.Conv3d",
            "maxpool2d": "nn.MaxPool2d",
            "maxpool1d": "nn.MaxPool1d",
            "avgpool2d": "nn.AvgPool2d",
            "avgpool1d": "nn.AvgPool1d",
            "adaptive_avgpool2d": "nn.AdaptiveAvgPool2d",
            "adaptive_avgpool1d": "nn.AdaptiveAvgPool1d",
            "linear": "nn.Linear",
            "batchnorm2d": "nn.BatchNorm2d",
            "batchnorm1d": "nn.BatchNorm1d",
            "batchnorm3d": "nn.BatchNorm3d",
            "layernorm": "nn.LayerNorm",
            "relu": "nn.ReLU",
            "leakyrelu": "nn.LeakyReLU",
            "prelu": "nn.PReLU",
            "elu": "nn.ELU",
            "gelu": "nn.GELU",
            "sigmoid": "nn.Sigmoid",
            "tanh": "nn.Tanh",
            "softmax": "nn.Softmax",
            "logsoftmax": "nn.LogSoftmax",
            "dropout": "nn.Dropout",
            "dropout2d": "nn.Dropout2d",
            "dropout3d": "nn.Dropout3d",
            "flatten": "nn.Flatten",
            "embedding": "nn.Embedding",
            "multiheadattention": "nn.MultiheadAttention",
            "transformerencoderlayer": "nn.TransformerEncoderLayer",
            "transformerencoder": "nn.TransformerEncoder",
            "lstm": "nn.LSTM",
            "gru": "nn.GRU",
            "rnn": "nn.RNN",
            "upsample": "nn.Upsample",
            "convtranspose2d": "nn.ConvTranspose2d",
            "convtranspose1d": "nn.ConvTranspose1d",
            "pixelshuffle": "nn.PixelShuffle",
            "groupnorm": "nn.GroupNorm",
            "instancenorm2d": "nn.InstanceNorm2d",
            "silu": "nn.SiLU",
            "mish": "nn.Mish",
            "hardswish": "nn.Hardswish",
            "input": "Input",
            "output": "Output",
            "add": "Add",
            "concat": "Concat",
            "reshape": "Reshape",
            "view": "View",
        }
        return type_map.get(layer_type.lower(), f"nn.{layer_type}")

    def _format_params(self, params: dict[str, Any]) -> str:
        param_strs = []
        for key, value in params.items():
            if isinstance(value, str):
                param_strs.append(f"{key}='{value}'")
            elif isinstance(value, bool):
                param_strs.append(f"{key}={str(value).capitalize()}")
            elif isinstance(value, list):
                param_strs.append(f"{key}={value}")
            elif isinstance(value, tuple):
                param_strs.append(f"{key}={value}")
            else:
                param_strs.append(f"{key}={value}")
        return ", ".join(param_strs)

    def _generate_layer_init(self, layer: LayerConfig) -> str:
        layer_type = layer.type.lower()
        
        if layer_type in ["input", "output"]:
            return None
        
        if layer_type in ["relu", "leakyrelu", "elu", "gelu", "sigmoid", "tanh", 
                          "softmax", "logsoftmax", "flatten", "silu", "mish", "hardswish",
                          "batchnorm2d", "batchnorm1d", "batchnorm3d"]:
            params_str = self._format_params(layer.params) if layer.params else ""
            return f"{self._get_layer_type(layer_type)}({params_str})"
        
        if layer_type == "dropout":
            p = layer.params.get("p", 0.5)
            return f"nn.Dropout(p={p})"
        
        if layer_type == "dropout2d":
            p = layer.params.get("p", 0.5)
            return f"nn.Dropout2d(p={p})"
        
        if layer_type == "conv2d":
            in_channels = layer.params.get("in_channels", 3)
            out_channels = layer.params.get("out_channels", 64)
            kernel_size = layer.params.get("kernel_size", 3)
            stride = layer.params.get("stride", 1)
            padding = layer.params.get("padding", 1)
            dilation = layer.params.get("dilation", 1)
            groups = layer.params.get("groups", 1)
            bias = layer.params.get("bias", True)
            return (f"nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, "
                    f"stride={stride}, padding={padding}, dilation={dilation}, "
                    f"groups={groups}, bias={str(bias).capitalize()})")
        
        if layer_type == "conv1d":
            in_channels = layer.params.get("in_channels", 64)
            out_channels = layer.params.get("out_channels", 128)
            kernel_size = layer.params.get("kernel_size", 3)
            stride = layer.params.get("stride", 1)
            padding = layer.params.get("padding", 1)
            return (f"nn.Conv1d({in_channels}, {out_channels}, kernel_size={kernel_size}, "
                    f"stride={stride}, padding={padding})")
        
        if layer_type == "maxpool2d":
            kernel_size = layer.params.get("kernel_size", 2)
            stride = layer.params.get("stride", 2)
            padding = layer.params.get("padding", 0)
            return f"nn.MaxPool2d(kernel_size={kernel_size}, stride={stride}, padding={padding})"
        
        if layer_type == "avgpool2d":
            kernel_size = layer.params.get("kernel_size", 2)
            stride = layer.params.get("stride", 2)
            return f"nn.AvgPool2d(kernel_size={kernel_size}, stride={stride})"
        
        if layer_type == "adaptive_avgpool2d":
            output_size = layer.params.get("output_size", 1)
            return f"nn.AdaptiveAvgPool2d(output_size={output_size})"
        
        if layer_type == "linear":
            in_features = layer.params.get("in_features", 512)
            out_features = layer.params.get("out_features", 10)
            bias = layer.params.get("bias", True)
            return f"nn.Linear({in_features}, {out_features}, bias={str(bias).capitalize()})"
        
        if layer_type == "layernorm":
            normalized_shape = layer.params.get("normalized_shape", 128)
            eps = layer.params.get("eps", 1e-05)
            return f"nn.LayerNorm({normalized_shape}, eps={eps})"
        
        if layer_type == "embedding":
            num_embeddings = layer.params.get("num_embeddings", 1000)
            embedding_dim = layer.params.get("embedding_dim", 128)
            return f"nn.Embedding({num_embeddings}, {embedding_dim})"
        
        if layer_type == "multiheadattention":
            embed_dim = layer.params.get("embed_dim", 128)
            num_heads = layer.params.get("num_heads", 8)
            dropout = layer.params.get("dropout", 0.1)
            return (f"nn.MultiheadAttention(embed_dim={embed_dim}, "
                    f"num_heads={num_heads}, dropout={dropout}, batch_first=True)")
        
        if layer_type == "transformerencoderlayer":
            d_model = layer.params.get("d_model", 128)
            nhead = layer.params.get("nhead", 8)
            dim_feedforward = layer.params.get("dim_feedforward", 512)
            dropout = layer.params.get("dropout", 0.1)
            activation = layer.params.get("activation", "relu")
            return (f"nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, "
                    f"dim_feedforward={dim_feedforward}, dropout={dropout}, "
                    f"activation='{activation}', batch_first=True)")
        
        if layer_type == "transformerencoder":
            num_layers = layer.params.get("num_layers", 2)
            return f"nn.TransformerEncoder(encoder_layer, num_layers={num_layers})"
        
        if layer_type == "lstm":
            input_size = layer.params.get("input_size", 128)
            hidden_size = layer.params.get("hidden_size", 256)
            num_layers = layer.params.get("num_layers", 1)
            dropout = layer.params.get("dropout", 0.0)
            bidirectional = layer.params.get("bidirectional", False)
            return (f"nn.LSTM({input_size}, {hidden_size}, num_layers={num_layers}, "
                    f"dropout={dropout}, bidirectional={str(bidirectional).capitalize()}, "
                    f"batch_first=True)")
        
        if layer_type == "gru":
            input_size = layer.params.get("input_size", 128)
            hidden_size = layer.params.get("hidden_size", 256)
            num_layers = layer.params.get("num_layers", 1)
            dropout = layer.params.get("dropout", 0.0)
            bidirectional = layer.params.get("bidirectional", False)
            return (f"nn.GRU({input_size}, {hidden_size}, num_layers={num_layers}, "
                    f"dropout={dropout}, bidirectional={str(bidirectional).capitalize()}, "
                    f"batch_first=True)")
        
        if layer_type == "upsample":
            size = layer.params.get("size", None)
            scale_factor = layer.params.get("scale_factor", 2)
            mode = layer.params.get("mode", "nearest")
            if size:
                return f"nn.Upsample(size={size}, mode='{mode}')"
            return f"nn.Upsample(scale_factor={scale_factor}, mode='{mode}')"
        
        if layer_type == "convtranspose2d":
            in_channels = layer.params.get("in_channels", 64)
            out_channels = layer.params.get("out_channels", 32)
            kernel_size = layer.params.get("kernel_size", 3)
            stride = layer.params.get("stride", 2)
            padding = layer.params.get("padding", 1)
            output_padding = layer.params.get("output_padding", 1)
            return (f"nn.ConvTranspose2d({in_channels}, {out_channels}, "
                    f"kernel_size={kernel_size}, stride={stride}, "
                    f"padding={padding}, output_padding={output_padding})")
        
        if layer_type == "groupnorm":
            num_groups = layer.params.get("num_groups", 32)
            num_channels = layer.params.get("num_channels", 64)
            return f"nn.GroupNorm(num_groups={num_groups}, num_channels={num_channels})"
        
        if layer_type == "pixelshuffle":
            upscale_factor = layer.params.get("upscale_factor", 2)
            return f"nn.PixelShuffle(upscale_factor={upscale_factor})"
        
        if layer_type == "add":
            return "Add()"
        
        if layer_type == "concat":
            dim = layer.params.get("dim", 1)
            return f"Concat(dim={dim})"
        
        if layer_type == "view":
            shape = layer.params.get("shape", [-1])
            return f"View({shape})"
        
        params_str = self._format_params(layer.params) if layer.params else ""
        return f"{self._get_layer_type(layer_type)}({params_str})"

    def _generate_forward_call(self, layer: LayerConfig, input_var: str) -> str:
        layer_type = layer.type.lower()
        
        if layer_type == "input":
            return input_var
        
        if layer_type == "output":
            return f"return {input_var}"
        
        if layer_type in ["conv2d", "conv1d", "linear", "batchnorm2d", "batchnorm1d",
                          "layernorm", "embedding", "upsample", "convtranspose2d",
                          "groupnorm", "instancenorm2d", "conv3d", "batchnorm3d",
                          "convtranspose1d", "maxpool2d", "maxpool1d", "avgpool2d",
                          "avgpool1d", "adaptive_avgpool2d", "adaptive_avgpool1d"]:
            return f"x = self.{layer.id}({input_var})"
        
        if layer_type in ["relu", "leakyrelu", "elu", "gelu", "sigmoid", "tanh",
                          "softmax", "logsoftmax", "dropout", "dropout2d", "dropout3d",
                          "flatten", "silu", "mish", "hardswish", "prelu"]:
            return f"x = self.{layer.id}({input_var})"
        
        if layer_type == "multiheadattention":
            return f"x, _ = self.{layer.id}({input_var}, {input_var}, {input_var})"
        
        if layer_type == "transformerencoderlayer":
            return f"x = self.{layer.id}({input_var})"
        
        if layer_type == "transformerencoder":
            return f"x = self.{layer.id}({input_var})"
        
        if layer_type in ["lstm", "gru", "rnn"]:
            return f"x, _ = self.{layer.id}({input_var})"
        
        if layer_type == "add":
            return f"x = {input_var}"
        
        if layer_type == "concat":
            return f"x = {input_var}"
        
        if layer_type == "view":
            shape = layer.params.get("shape", [-1])
            shape_str = ", ".join(str(s) for s in shape)
            return f"x = {input_var}.view({shape_str})"
        
        return f"x = self.{layer.id}({input_var})"

    def generate(self) -> str:
        self.validate()
        
        model_name = self.blueprint.model_name
        use_jit = self.blueprint.use_jit
        use_compile = self.blueprint.use_compile
        
        layers_by_id = {layer.id: layer for layer in self.blueprint.layers}
        
        sorted_layers = self._topological_sort()
        
        init_lines = []
        forward_lines = []
        
        has_transformer_encoder = any(l.type.lower() == "transformerencoder" for l in self.blueprint.layers)
        has_transformer_encoder_layer = any(l.type.lower() == "transformerencoderlayer" for l in self.blueprint.layers)
        
        if has_transformer_encoder and has_transformer_encoder_layer:
            encoder_layer = None
            encoder = None
            for layer in self.blueprint.layers:
                if layer.type.lower() == "transformerencoderlayer":
                    encoder_layer = layer
                if layer.type.lower() == "transformerencoder":
                    encoder = layer
            
            if encoder_layer and encoder:
                d_model = encoder_layer.params.get("d_model", 128)
                nhead = encoder_layer.params.get("nhead", 8)
                dim_feedforward = encoder_layer.params.get("dim_feedforward", 512)
                dropout = encoder_layer.params.get("dropout", 0.1)
                activation = encoder_layer.params.get("activation", "relu")
                num_layers = encoder.params.get("num_layers", 2)
                
                init_lines.append(
                    f"        encoder_layer = nn.TransformerEncoderLayer("
                    f"d_model={d_model}, nhead={nhead}, "
                    f"dim_feedforward={dim_feedforward}, dropout={dropout}, "
                    f"activation='{activation}', batch_first=True)"
                )
                init_lines.append(
                    f"        self.transformer = nn.TransformerEncoder("
                    f"encoder_layer, num_layers={num_layers})"
                )
        
        for layer in sorted_layers:
            layer_type = layer.type.lower()
            
            if layer_type in ["input", "output"]:
                continue
            
            if layer_type == "transformerencoder" and has_transformer_encoder_layer:
                continue
            
            layer_init = self._generate_layer_init(layer)
            if layer_init:
                init_lines.append(f"        self.{layer.id} = {layer_init}")
        
        forward_lines.append("    def forward(self, x):")
        
        input_layers = [l for l in sorted_layers if l.type.lower() == "input"]
        if not input_layers:
            forward_lines.append("        # Input")
        else:
            forward_lines.append(f"        # Input: {input_layers[0].id}")
        
        prev_layer = None
        for layer in sorted_layers:
            layer_type = layer.type.lower()
            
            if layer_type == "input":
                continue
            
            if layer_type == "output":
                forward_lines.append(f"        return x")
                continue
            
            if prev_layer and prev_layer.type.lower() not in ["input", "output"]:
                forward_lines.append(f"        # {layer.type.upper()}")
            elif prev_layer and prev_layer.type.lower() == "input":
                forward_lines.append(f"        # {layer.type.upper()}")
            
            forward_call = self._generate_forward_call(layer, "x")
            if forward_call:
                forward_lines.append(f"        {forward_call}")
            
            prev_layer = layer
        
        has_return = any("return x" in line for line in forward_lines)
        if not has_return:
            forward_lines.append("        return x")
        
        code = []
        code.append("import torch")
        code.append("import torch.nn as nn")
        code.append("import torch.nn.functional as F")
        code.append("")
        code.append("")
        code.append("class Add(nn.Module):")
        code.append("    def __init__(self):")
        code.append("        super().__init__()")
        code.append("")
        code.append("    def forward(self, x):")
        code.append("        return x + x")
        code.append("")
        code.append("")
        code.append("class Concat(nn.Module):")
        code.append("    def __init__(self, dim=1):")
        code.append("        super().__init__()")
        code.append("        self.dim = dim")
        code.append("")
        code.append("    def forward(self, x):")
        code.append("        return torch.cat([x, x], dim=self.dim)")
        code.append("")
        code.append("")
        code.append("class View(nn.Module):")
        code.append("    def __init__(self, shape):")
        code.append("        super().__init__()")
        code.append("        self.shape = shape")
        code.append("")
        code.append("    def forward(self, x):")
        code.append("        return x.view(self.shape)")
        code.append("")
        code.append("")
        code.append(f"class {model_name}(nn.Module):")
        code.append("    def __init__(self):")
        code.append("        super().__init__()")
        
        if init_lines:
            code.extend(init_lines)
        else:
            code.append("        pass")
        
        code.append("")
        code.extend(forward_lines)
        code.append("")
        code.append("")
        
        if use_jit:
            code.append(f"# JIT Traced Model")
            code.append(f"def create_{model_name.lower()}_jit():")
            code.append(f"    model = {model_name}()")
            code.append(f"    model.eval()")
            code.append(f"    example_input = torch.randn(1, 3, 224, 224)")
            code.append(f"    traced_model = torch.jit.trace(model, example_input)")
            code.append(f"    return traced_model")
            code.append("")
        
        if use_compile:
            code.append(f"# torch.compile Model (PyTorch 2.0+)")
            code.append(f"def create_{model_name.lower()}_compiled():")
            code.append(f"    model = {model_name}()")
            code.append(f"    compiled_model = torch.compile(model)")
            code.append(f"    return compiled_model")
            code.append("")
        
        code.append("")
        code.append(f"if __name__ == '__main__':")
        code.append(f"    model = {model_name}()")
        code.append(f"    print(model)")
        code.append(f"    print(f'Total parameters: {{sum(p.numel() for p in model.parameters()):,}}')")
        code.append(f"")
        code.append(f"    # Test forward pass")
        code.append(f"    x = torch.randn(1, 3, 224, 224)")
        code.append(f"    output = model(x)")
        code.append(f"    print(f'Input shape: {{x.shape}}')")
        code.append(f"    print(f'Output shape: {{output.shape}}')")
        
        return "\n".join(code)

    def _topological_sort(self) -> list[LayerConfig]:
        layers_by_id = {layer.id: layer for layer in self.blueprint.layers}
        
        adj = {layer.id: [] for layer in self.blueprint.layers}
        in_degree = {layer.id: 0 for layer in self.blueprint.layers}
        
        for conn in self.blueprint.connections:
            if conn.from_id in adj and conn.to_id in in_degree:
                adj[conn.from_id].append(conn.to_id)
                in_degree[conn.to_id] += 1
        
        queue = []
        for layer in self.blueprint.layers:
            if in_degree[layer.id] == 0:
                queue.append(layer.id)
        
        queue.sort(key=lambda x: layers_by_id[x].position.get("x", 0))
        
        result = []
        while queue:
            node = queue.pop(0)
            result.append(layers_by_id[node])
            
            neighbors = sorted(adj[node], key=lambda x: layers_by_id[x].position.get("x", 0))
            for neighbor in neighbors:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.blueprint.layers):
            result = sorted(self.blueprint.layers, key=lambda x: x.position.get("x", 0))
        
        return result
