# Research Sprint Schedule (11 Weeks)

**Target**: NeurIPS 2026 Submission
**Buffer**: 1 Week (Week 12)

## Phase 1: Engine & Benchmarks (Weeks 1-4)
*Focus: Code stability and performance data collection.*

*   **Week 1: The Pruning Algorithm** 🟢
    *   Refine regex for Section splitting (Method/Conclusion).
    *   Implement "Content-Aware" page selection logic.
    *   *Deliverable*: `src/core/pruner.py` with >95% section retrieval accuracy.
*   **Week 2: Baselines on Server (3090)** 🟢
    *   Benchmark `MinerU` (Full) and `Flash-MinerU` on 3090.
    *   Validate "Ours" logic (ONNX Runtime simulation on 3090).
    *   *Deliverable*: Preliminary speed comparison table.
*   **Week 3: Edge Migration (The Hardest Part)** 🔴
    *   Convert ONNX to TensorRT Engines on AGX Orin.
    *   Attempt DLA mapping (`--useDLACore=0`). **Fallback Strategy**: Use GPU FP16 if DLA fails.
    *   *Deliverable*: Working `scripts/deploy_jetson.py` and `src/inference/dla_engine.py`.
*   **Week 4: Throughput Stress Test** 🟡
    *   Run batch processing (100+ papers) on AGX.
    *   Measure `jtop` metrics (Power, GPU Load).
    *   *Deliverable*: Throughput charts for the paper (Figure 3 & 4).

## Phase 2: Application & Eval (Weeks 5-7)
*Focus: Data utility and downstream quality.*

*   **Week 5: Database & Storage** 🟢
    *   Setup SQLite schema.
    *   Implement bulk ingestion script (`src/database/rdbms_handler.py`).
*   **Week 6: Level 2 Interaction (RAG)** 🟡
    *   Implement "Multi-paper Comparison" prompt logic.
    *   Verify if Qwen can understand the OCR-reconstructed tables.
*   **Week 7: Accuracy Evaluation** 🟡
    *   Manual QA test: Compare "Ours" answers vs "Full Text" answers.
    *   *Deliverable*: "Accuracy Preservation" table (Table 2).

## Phase 3: Writing (Weeks 8-11)
*Focus: Storytelling and Polishing.*

*   **Week 8: Methodology Section**
    *   Draft System Architecture.
    *   Create the main architectural diagram.
    *   *Focus*: Describe "Content-Aware Pruning" algorithm.
*   **Week 9: Experiments Section**
    *   Visualize Week 4 and Week 7 data.
    *   Write "Ablation Study" (Effect of Pruning vs. Effect of DLA).
*   **Week 10: Intro & Related Work**
    *   Position paper against MinerU, LayoutLM, etc.
*   **Week 11: Final Polish**
    *   Abstract refinement.
    *   Formatting check (BibTeX, Templates).