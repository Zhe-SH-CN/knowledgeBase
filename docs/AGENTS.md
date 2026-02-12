# AGENTS.md - Codebase Guidelines for AI Agents

## Overview

This repository contains a paper processing system with OCR and LLM capabilities. Key components:
- `main.py` - Paper metadata extraction using HuggingFace transformers
- `main-llamacpp.py` - Local LLM inference using llama.cpp
- `test-ocr.py` - PaddleOCR document parsing experiments
- `PaddleOCR/` - OCR library submodule

## Build, Lint, and Test Commands

### Virtual Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Run Python scripts
python main.py
python main-llamacpp.py
python test-ocr.py
```

### Running Tests
```bash
# Run all tests
pytest

# Run single test file
pytest test-ocr.py

# Run single test function
pytest test-ocr.py::FullOCRExperiment::run -v

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Skip resource-intensive tests
pytest -m 'not resource_intensive'
```

### Code Quality
```bash
# Lint with ruff (already configured)
ruff check .

# Format with ruff
ruff format .

# Run pre-commit hooks (PaddleOCR submodule)
pre-commit run --all-files

# Black formatting (PaddleOCR style)
black .
```

### PaddleOCR Specific
```bash
# Install PaddleOCR
cd PaddleOCR && pip install -e .

# Install with all dependencies
pip install -e ".[all]"

# Run PaddleOCR tests
pytest PaddleOCR/
```

## Code Style Guidelines

### Imports
- Standard library imports first, sorted alphabetically
- Third-party imports second, sorted alphabetically
- Blank line between standard library and third-party imports
- No unused imports

```python
# Correct
import json
import os
import time
from pathlib import Path

import fitz
import torch
from tqdm import tqdm
```

### Formatting
- Line length: 100 characters (ruff default, aligns with Black)
- Use 4 spaces for indentation (no tabs)
- No trailing whitespace
- Blank lines: two between top-level definitions, one between method definitions

### Type Hints
- Use type hints for function parameters and return values
- Use `str`, `int`, `float`, `bool` for simple types
- Use `List[T]`, `Dict[K, V]`, `Optional[T]` for generics
- Use `Path` from pathlib instead of strings for file paths

```python
def process_paper(self, pdf_path: str, pbar: tqdm) -> Dict[str, Any]:
    ...
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `PaperMetadataGenerator`, `LlamaNanoManager`)
- **Functions/Variables**: snake_case (e.g., `extract_text_from_pdf`, `api_keys`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `INPUT_DIR`, `METADATA_DIR`)
- **Private Methods**: prefix with underscore (e.g., `_extract_metadata`)
- **Descriptive names**: Avoid single-letter names except for counters

### Error Handling
- Use try/except blocks with specific exception types
- Log errors with descriptive messages
- Handle file I/O errors gracefully
- Return sensible defaults or raise custom exceptions

```python
try:
    metadata = json.loads(result_text)
except json.JSONDecodeError as e:
    metadata = {
        "title": "未能提取",
        "authors": [],
        "abstract": "",
        "keywords": [],
        "category": "未分类",
        "key_contributions": [],
    }
```

### File Operations
- Use `os.makedirs(directory, exist_ok=True)` for directory creation
- Use `encoding="utf-8"` for text file operations
- Use `ensure_ascii=False, indent=2` for JSON dumps
- Close resources properly (or use context managers)

### Project-Specific Patterns

#### Configuration Section
```python
# ================= 配置区 =================
INPUT_DIR = "./osdi2025"
OUTPUT_BASE_DIR = "./paper_states"
```

#### Class Structure
- `__init__` for initialization and device setup
- `process_*` methods for main operations
- `get_*` or `calculate_*` for data retrieval
- `save_*` for persistence
- Use progress bars (tqdm) for batch operations

#### JSON Output
- Use Chinese keys for metadata processing
- Include timestamps in output files
- Separate metadata and tags into different files

#### GPU/CUDA Handling
```python
if torch.cuda.is_available():
    self.device = torch.device("cuda:0")
else:
    self.device = torch.device("cpu")
```

### Documentation
- Use docstrings for public classes and methods
- Include parameter and return type descriptions
- Use Chinese comments for configuration explanations
- Document complex logic inline

### Security
- Never commit API keys or credentials
- Load sensitive data from files (e.g., `siliconflow-apikeys.txt`)
- Use `.gitignore` for sensitive files
