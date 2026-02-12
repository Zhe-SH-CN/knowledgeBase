# EdgeScholar: High-Throughput Academic Knowledge Construction on the Edge

[English](README.md) | [简体中文](docs/README.zh-CN.md)

![Status](https://img.shields.io/badge/Status-Research_Preview-blue) ![Hardware](https://img.shields.io/badge/Hardware-Jetson_AGX_Orin-green) ![Target](https://img.shields.io/badge/Target-NeurIPS_2026-red)

**EdgeScholar** is a privacy-first, high-throughput academic paper parsing and knowledge retrieval system designed for resource-constrained edge devices (specifically NVIDIA Jetson AGX Orin).

Unlike cloud-based solutions (e.g., MinerU, GPT-4) that prioritize rendering perfection at the cost of latency and privacy, EdgeScholar focuses on **Content-Aware Pruning** and **Heterogeneous Computing** to maximize **Papers-Per-Hour** throughput while maintaining semantic fidelity.

## 🚀 Key Features

*   **⚡ Extreme Throughput**: Optimized for offline indexing of thousands of papers. Achieves 10x-20x speedup over full OCR pipelines.
*   **🧠 Content-Aware Pruning**: Automatically identifies high-value pages (Methodology, Experiments) and skips redundant content (References, Related Work) to save compute.
*   **⚙️ Heterogeneous Acceleration**:
    *   **CPU**: Text extraction (PyMuPDF) & Recognition (SVTR-Tiny).
    *   **DLA (Deep Learning Accelerator)**: Layout Analysis (DocLayout-YOLO) & Text Detection (DBNet++).
    *   **GPU**: Semantic Synthesis (Qwen2.5-3B).
*   **🔒 Privacy First**: 100% On-Device execution. No data leaves your local network.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **Hardware**: NVIDIA Jetson AGX Orin (JetPack 5.x) / RTX 3090 (Dev)
*   **Models**:
    *   Layout: `DocLayout-YOLO` (ONNX/TensorRT)
    *   OCR Det: `DBNet++` (ONNX/TensorRT)
    *   OCR Rec: `SVTR-Tiny` (ONNX Runtime)
    *   LLM: `Qwen2.5-3B-Instruct` (Transformers/FP16)
*   **Core Libs**: `PyMuPDF`, `RapidOCR-ONNXRuntime`, `Ultralytics`

## 📊 Performance (Target)

| Method | Avg. Latency (s/paper) | Throughput (papers/min) | Resource |
| :--- | :---: | :---: | :--- |
| PyMuPDF Only | 0.2s | ~300 | CPU Only |
| MinerU (Full) | ~35.0s | ~1.7 | Heavy GPU |
| **EdgeScholar (Ours)** | **~2.5s** | **~24** | **Hybrid (CPU+DLA+GPU)** |

## 📦 Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/EdgeScholar.git
cd EdgeScholar

# 2. Install dependencies (uv recommended)
uv pip install pymupdf opencv-python numpy ultralytics rapidocr-onnxruntime

# 3. Model Preparation
# See scripts/download_models.sh for ONNX links
```

## 📂 Project Structure

```
EdgeScholar/
├── src/                    # Source code
│   └── test/              # Test modules (more modules coming soon)
├── models/                # ONNX/TensorRT model weights
├── input/                 # Input academic papers (ignored by git)
├── output/                # Processing results & benchmarks (ignored by git)
├── docs/                  # Documentation
│   ├── README.zh-CN.md   # Simplified Chinese documentation
│   ├── ARCHITECTURE.md
│   ├── PRD.md
│   ├── SCHEDULE.md
│   ├── AGENTS.md
│   └── pdf2md.md
├── .gitignore
├── README.md
└── ...
```

## 🗓️ Roadmap

- [ ] Level 1: Content-Aware Parsing & Structured Metadata Extraction.

- [ ] Level 2: Local RAG & Multi-paper Summarization with SQLite.

- [ ] Level 3: Idea Evolution Graph & Novelty Detection.