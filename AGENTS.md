# AGENTS.md - NNFactory Development Guide

## Project Overview

NNFactory is a visual neural network blueprint maker with a PyTorch backend. Users drag-and-drop layer components on an HTML5 canvas to design architectures, then export optimized PyTorch code with JIT and torch.compile support.

**Stack**: FastAPI (Python backend) + Vanilla HTML5/CSS3/JS (frontend)

## Project Structure

```
NNFactory/
├── backend/
│   ├── main.py              # FastAPI server, Pydantic models, routes
│   ├── code_generator.py    # Blueprint-to-PyTorch code generation engine
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Main HTML entry point
│   ├── css/style.css        # All styles (CSS custom properties, dark theme)
│   └── js/
│       ├── app.js           # Application entry, event orchestration
│       ├── canvas.js        # CanvasManager: zoom, pan, grid rendering
│       ├── nodes.js         # NodeManager: drag-drop, create, delete nodes
│       ├── connections.js   # ConnectionManager: bezier curve connections
│       ├── properties.js    # PropertiesPanel: layer parameter editing
│       └── codegen.js       # CodeGenerator: frontend fallback code gen
├── start.bat                # Windows one-click startup
└── README.md
```

## Commands

### Backend (Python/FastAPI)
```bash
# Start dev server (auto-reload)
cd backend && uvicorn main:app --reload --port 8000

# Install dependencies
cd backend && pip install -r requirements.txt

# Run API docs (auto-available at)
# http://localhost:8000/docs

# Quick test - import validation
cd backend && python -c "from code_generator import CodeGenerator; print('OK')"

# Test code generation
cd backend && python -c "
from code_generator import CodeGenerator
from main import Blueprint, LayerConfig, Connection
# ... build blueprint and call gen.generate()
"
```

### Frontend (Vanilla JS)
```bash
# Start static file server
cd frontend && python -m http.server 3000

# Access at http://localhost:3000
```

### One-Click Start (Windows)
```bash
start.bat
```

### Testing
No formal test framework exists. Test manually:
1. Start both servers (`start.bat`)
2. Open http://localhost:3000
3. Drag layers onto canvas, connect them, click "Generate Code"
4. Verify generated Python code is syntactically valid

To validate generated code programmatically:
```bash
python -c "import ast; ast.parse(open('model.py').read()); print('Valid Python')"
```

## Code Style Guidelines

### Python (Backend)

**Imports**: Standard library first, then third-party, then local. One import per line.
```python
from fastapi import FastAPI
from pydantic import BaseModel
from code_generator import CodeGenerator
```

**Types**: Use modern Python type hints (`list[X]`, `dict[K, V]` not `List[X]`). All function signatures must be typed. Use `Any` from `typing` for dynamic params.

**Naming**: `snake_case` for functions/variables, `PascalCase` for classes. Private methods prefixed with `_`.

**Formatting**: 4-space indentation. Max line length ~120 chars. Use f-strings exclusively.

**Error Handling**: Raise `ValueError` for validation errors. Catch broad `Exception` only at API boundaries. Return structured error responses `{"valid": False, "errors": [...]}`.

**Pydantic Models**: Define in `main.py` alongside routes. Use `dict[str, Any]` for flexible layer params.

**Code Generation**: Python booleans must be capitalized (`True`/`False`, not `true`/`false`). Use `str(value).capitalize()` for bool-to-string conversion.

### JavaScript (Frontend)

**Architecture**: Class-based modules. Each file exports one primary class. `window.app` is the global singleton for cross-module communication.

**Naming**: `camelCase` for methods/variables, `PascalCase` for classes. DOM IDs use `camelCase` (e.g., `generateBtn`).

**DOM Access**: Cache DOM references in constructor. Use `document.getElementById` for performance.

**Event Handling**: Use `addEventListener` (not inline handlers). Clean up listeners on teardown. Use event delegation where practical.

**CSS**: Use CSS custom properties (`--accent-primary`) for theming. BEM-like naming: `.component-element--modifier`.

**No frameworks**: Vanilla JS only. No jQuery, no build step.

### HTML

**Structure**: Semantic elements (`header`, `main`, `aside`). Inline SVG icons (no external icon libs).

**Accessibility**: Use `title` attributes on buttons. Labels for inputs.

## Key Patterns

**Node creation**: `nodeManager.createNode(type, x, y)` auto-snaps to grid and renders.

**Connections**: Drag from output port (right) to input port (left). Bezier curves with arrow markers.

**Code generation flow**: Frontend builds blueprint JSON → POST `/generate` → Backend `CodeGenerator` produces PyTorch code → Frontend displays in modal. Falls back to `codegen.js` if backend unavailable.

**Topological sort**: Layers are ordered by connections for correct `__init__`/`forward` generation. Falls back to x-position sort on cycles.

## Important Notes

- The frontend has a full code generation fallback in `codegen.js` — the backend is optional but preferred.
- Generated code includes helper classes (`Add`, `Concat`, `View`) for common operations.
- `TransformerEncoder` requires a paired `TransformerEncoderLayer` — the generator handles this specially.
- Boolean params in generated Python must use `True`/`False` (capitalized).
- The `bias` param defaults to `True` for Conv2d and Linear layers.
- Canvas uses `screenToWorld`/`worldToScreen` transforms for zoom/pan coordinate mapping.
- All node positions snap to a 24px grid.
