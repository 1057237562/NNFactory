# NNFactory - Neural Network Blueprint Maker

A visual drag-and-drop neural network builder with PyTorch code generation. Design architectures visually, then export optimized PyTorch code with JIT & torch.compile support.

## Features

- **Drag & Drop Interface**: Build neural networks by dragging layer components onto a visual canvas
- **30+ Layer Types**: Conv2d, Linear, LSTM, Transformer, BatchNorm, activations, pooling, and more
- **Visual Connections**: Connect layers with bezier curves to define data flow
- **Property Editor**: Configure layer parameters (channels, kernel sizes, etc.) in real-time
- **Code Generation**: Convert blueprints to optimized PyTorch code
- **JIT Support**: Generate torch.jit.trace compatible models
- **torch.compile**: Generate torch.compile ready code (PyTorch 2.0+)
- **Export/Import**: Save and load blueprints as JSON
- **Zoom & Pan**: Navigate large architectures with ease
- **Dark Theme**: Professional dark UI for extended use

## Quick Start

### Option 1: One-Click Start (Windows)
```bash
start.bat
```

### Option 2: Manual Start

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
python -m http.server 4000
```

Open http://localhost:4000 in your browser.

## Usage

### Building a Network

1. **Drag layers** from the left sidebar onto the canvas
2. **Connect layers** by dragging from output ports (right) to input ports (left)
3. **Configure parameters** by clicking on a node to open the properties panel
4. **Generate code** by clicking the "Generate Code" button

### Supported Layer Types

| Category | Layers |
|----------|--------|
| **Convolution** | Conv2d, Conv1d, ConvTranspose2d |
| **Pooling** | MaxPool2d, AvgPool2d, AdaptiveAvgPool2d |
| **Linear** | Linear (Dense), Embedding |
| **Normalization** | BatchNorm2d, BatchNorm1d, LayerNorm, GroupNorm |
| **Activation** | ReLU, LeakyReLU, GELU, Sigmoid, Tanh, Softmax, SiLU |
| **Transformer** | MultiheadAttention, TransformerEncoderLayer |
| **Recurrent** | LSTM, GRU |
| **Regularization** | Dropout, Dropout2d |
| **Utility** | Input, Flatten, Upsample, PixelShuffle |

### Export Options

- **Generate Code**: Opens modal with PyTorch code (copy or download)
- **Export Blueprint**: Save architecture as JSON for later editing
- **Import Blueprint**: Load a previously saved JSON blueprint

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Delete` / `Backspace` | Delete selected node or connection |
| `Escape` | Close modal / deselect |
| `Alt + Drag` | Pan canvas |
| `Scroll` | Zoom in/out |

## Project Structure

```
NNFactory/
├── backend/
│   ├── main.py                  # FastAPI server, Pydantic models, routes
│   ├── code_generator.py        # PyTorch code generation engine
│   ├── device_utils.py          # Hardware backend detection
│   ├── evaluator.py             # Model evaluation logic
│   ├── requirements.txt         # Python dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   └── blueprint.py         # Blueprint parsing & validation
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── info.py              # Dataset metadata extraction
│   │   ├── manager.py           # Dataset CRUD operations
│   │   └── stats.py             # Dataset statistics
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataloader.py        # Data loading utilities
│   │   └── engine.py            # Training loop engine
│   └── preprocessing/
│       ├── __init__.py
│       ├── operations.py        # Preprocessing operations
│       └── pipeline.py          # Pipeline execution engine
├── frontend/
│   ├── index.html               # Main HTML entry point
│   ├── css/
│   │   ├── base.css             # CSS custom properties, reset, body defaults
│   │   ├── layout.css           # App shell, header, sidebar, properties panel
│   │   ├── components.css       # Buttons, toggles, inputs, toasts, forms
│   │   ├── canvas.css           # Canvas container, nodes, connections
│   │   ├── modals.css           # Modal overlays, code modals
│   │   ├── training.css         # Training config, progress, status bar
│   │   ├── evaluator.css        # Evaluation UI, metrics, predictions
│   │   ├── dataset-manager.css  # Dataset manager base styles
│   │   ├── dataset-manager-pagination.css  # Dataset lists, previews, image grids
│   │   ├── dataset-manager-buttons.css     # Dataset button/toolbar/legacy styles
│   │   ├── dataset-viz.css      # Dataset visualization stats, bar charts, tags
│   │   ├── dataset-viz-charts.css          # Chart-specific viz (relations, dist bars)
│   │   ├── preprocessing.css    # Preprocessing pipeline blueprint
│   │   └── weights.css          # Weights list and management
│   ├── html/
│   │   ├── sidebar-layers.html  # Sidebar layer category template
│   │   └── modals/
│   │       ├── code-modal.html
│   │       ├── train-modal.html
│   │       ├── eval-modal.html
│   │       ├── dataset-modal.html
│   │       ├── weights-modal.html
│   │       ├── pp-node-modal.html
│   │       └── pp-result-modal.html
│   └── js/
│       ├── app.js               # Application entry, event orchestration
│       ├── canvas.js            # CanvasManager: zoom, pan, grid rendering
│       ├── nodes.js             # NodeManager: drag-drop, create, delete nodes
│       ├── connections.js       # ConnectionManager: bezier curve connections
│       ├── properties.js        # PropertiesPanel: layer parameter editing
│       ├── codegen.js           # CodeGenerator: frontend fallback code gen
│       ├── utils.js             # Shared utility functions
│       ├── path-utils.js        # Blueprint path extraction utilities
│       ├── layer-config.js      # Layer type configuration definitions
│       ├── modal-loader.js      # Dynamic modal HTML loading
│       ├── training-chart.js    # Training loss/accuracy charting
│       ├── training-ui.js       # Training modal UI and controls
│       ├── weights-ui.js        # Weights modal UI and management
│       ├── evaluator-ui.js      # Evaluation UI and prediction display
│       └── dataset/
│           ├── dataset-manager.js  # Dataset manager UI logic
│           ├── dataset-viz.js      # Dataset visualization UI
│           └── pp-blueprint.js     # Preprocessing blueprint editor
├── start.bat                  # Windows one-click startup
└── README.md
```

## API Endpoints

- `POST /generate` - Generate PyTorch code from blueprint
- `POST /validate` - Validate blueprint structure
- `GET /health` - Health check

## Blueprint JSON Format

```json
{
  "layers": [
    {
      "id": "node_1",
      "type": "conv2d",
      "params": {
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
      },
      "position": { "x": 100, "y": 200 }
    }
  ],
  "connections": [
    { "from_id": "node_1", "to_id": "node_2" }
  ],
  "model_name": "NeuralNetwork",
  "use_jit": false,
  "use_compile": true
}
```

## Tech Stack

- **Frontend**: Vanilla HTML5, CSS3, JavaScript (no frameworks)
- **Backend**: Python, FastAPI, Pydantic
- **Code Generation**: Custom engine producing idiomatic PyTorch code

## Backend Support

NNFactory supports multiple compute backends for model training and code generation. Hardware is auto-detected at runtime, and all backends gracefully fall back to CPU if hardware is unavailable.

| Backend | Device Type | Platform | Extra Requirements |
|---------|-------------|----------|-------------------|
| **NVIDIA CUDA** | `cuda` | Linux, Windows | PyTorch with CUDA (comes with standard PyTorch) |
| **AMD ROCm** | `rocm` | Linux | PyTorch with ROCm support (`pip install torch==<ver> --index-url https://download.pytorch.org/whl/rocm5.6`) |
| **Intel XPU** | `xpu` | Linux, Windows | `intel-extension-for-pytorch` (IPEX) — optional install |
| **Apple MPS** | `mps` | macOS 14.0+ (Apple Silicon) | Built into PyTorch on macOS |

### Backend Details

- **NVIDIA CUDA**: Default GPU backend. Detected via `torch.cuda.is_available()`. Works out of the box with standard PyTorch installations.
- **AMD ROCm**: Uses CUDA device interface (`torch.device('cuda')`) via AMD's HIP runtime. Detected via `torch.cuda.is_available() and torch.version.hip`. Linux-only.
- **Intel XPU**: Requires the optional `intel-extension-for-pytorch` package. Detected via `torch.xpu.is_available()`. Install with `pip install intel-extension-for-pytorch`.
- **Apple MPS**: Built into PyTorch on macOS. Detected via `torch.backends.mps.is_available()`. Requires macOS 14.0+ on Apple Silicon hardware.

### Generated Code

When you generate PyTorch code from a blueprint, the generated `__main__` block automatically includes device detection code for the selected backend:

```python
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # ROCm uses same pattern
    model = NeuralNetwork().to(device)
    # ...
```

This makes the generated code portable — it will use the available hardware wherever it runs.

### Device Selection

Use the device dropdown in the NNFactory toolbar to select your target backend:
- **CPU**: Always available, no GPU required
- **NVIDIA CUDA**: For NVIDIA GPUs
- **AMD ROCm**: For AMD GPUs on Linux
- **Intel XPU**: For Intel Arc, Flex, or Max GPUs (requires IPEX)
- **Apple MPS**: For Apple Silicon Macs (macOS 14.0+)

## License

MIT
