# System Architecture & Project Structure

## 1. Directory Structure

This structure is designed to decouple model inference (backends) from business logic (pipeline).

```text
EdgeScholar/
├── configs/
│   ├── models.yaml             # Model paths and hardware config (DLA/GPU IDs)
│   └── scraper_headers.json    # Headers for WebMeta scraper
├── data/
│   ├── input_pdfs/             # Raw PDF files
│   ├── output_vault/           # Processed JSON/Markdown results
│   └── database/               # SQLite db file
├── models/
│   ├── onnx/                   # ONNX weights (DocLayout, DBNet, SVTR)
│   ├── engines/                # TensorRT engines (Generated on Jetson)
│   └── llm/                    # Qwen2.5-3B GGUF or SafeTensors
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── pruner.py           # Content-Aware Pruning (PyMuPDF logic)
│   │   ├── scraper.py          # Metadata Scraper (CrossRef/arXiv/Web)
│   │   └── aggregator.py       # Prompt Assembly & Markdown generation
│   ├── inference/
│   │   ├── dla_engine.py       # Wrapper for TensorRT DLA execution (Layout/Det)
│   │   ├── cpu_ocr.py          # Wrapper for ONNXRuntime CPU execution (Rec)
│   │   └── llm_client.py       # Wrapper for Transformers/Llama.cpp (Qwen)
│   └── pipeline/
│       └── manager.py          # Main producer-consumer loop & ThreadPool
├── scripts/
│   ├── benchmark_3090.py       # Simulating heterogeneity on Server
│   ├── deploy_jetson.py        # TensorRT conversion script (trtexec wrapper)
│   └── run_batch.py            # Main entry point
├── tests/                      # Unit tests
└── requirements.txt
```
## 2. Heterogeneous Pipeline Design
The system employs a Producer-Consumer architecture to decouple parsing (CPU/DLA) from reasoning (GPU).

```Mermaid
graph TD
    subgraph "Stage 1: CPU I/O & Dispatch (Producer)"
    PDF[Raw PDF] --> PyMuPDF[PyMuPDF Scanner]
    PyMuPDF -->|Text Stream| TextFilter[Structure Pruner]
    PyMuPDF -->|Page Images| Router{Visual Router}
    TextFilter -->|Method/Conclusion| PromptBuffer
    end

    subgraph "Stage 2: DLA Visual Acceleration"
    Router -->|Visual Page| DLA0["Layout Analysis\nDocLayout-YOLO"]
    DLA0 -->|Table/Figure Crop| DLA1["Text Detection\nDBNet++"]
    end

    subgraph "Stage 3: CPU Parallel Rec (Auxiliary)"
    DLA1 -->|Text Patches| CPU_Rec["Recognition\nSVTR-Tiny (6-Cores)"]
    CPU_Rec -->|Structured Data| PromptBuffer
    end

    subgraph "Stage 4: GPU Semantic Synthesis (Consumer)"
    PromptBuffer -->|Context| Qwen["Qwen2.5-3B\n(FP16 + FlashAttn)"]
    Qwen -->|Summary/Tags| DB[("SQLite / JSON")]
    end
```

## 3. Component Details
#### A. Pre-processing & Routing (CPU)
- Logic: Scan full text to locate References (end point) and Methodology (start point).
- Pruning: Drop Related Work and References.
- Trigger: Only send pages containing keywords (Table, Figure) or image objects to the Vision Pipeline.
#### B. Vision Pipeline (DLA + CPU)
To save GPU memory for the LLM, vision tasks are offloaded:
- Layout: DocLayout-YOLO (TensorRT Engine on DLA Core 0).
- Detection: DBNet++ (TensorRT Engine on DLA Core 1).
- Recognition: SVTR-Tiny (ONNX Runtime on CPU).
- Reconstruction: A heuristic geometric algorithm (CPU) converts OCR coordinates into Markdown Tables.
#### C. Logic Core (GPU)
- **Model**: Qwen2.5-3B-Instruct.
- **Optimization**:
    - FP16 Precision.
    - torch.inference_mode().
    - sdpa kernel (Flash Attention).
- **Input**: Highly compressed text + Markdown Tables + Chart Captions/Legends.
