# Product Requirements Document (PRD)

## 1. Problem Definition
Current academic parsing tools face an "Impossible Triangle" on edge devices:
1.  **High Accuracy** (e.g., MinerU) -> Extremely slow (low throughput).
2.  **High Speed** (e.g., PyMuPDF) -> Loss of structure (tables/charts).
3.  **Privacy** (Local LLM) -> Resource contention between OCR and LLM.

**Goal**: Build a system that achieves **Maximum Throughput** (Papers/Hour) on Jetson AGX Orin for offline knowledge base construction, without sacrificing the semantic integrity of key experimental data (Tables/Figures).

## 2. Functional Requirements

### Level 1: The Parsing Engine (Core)
*   **FR-1.1**: Must extract metadata (Title, Authors, Year) with >98% accuracy using a hybrid approach (Scraper > LLM > Rules).
*   **FR-1.2**: Must identify and extract structure from Tables and Figures into Markdown/Text description.
*   **FR-1.3**: Must implement **Content-Aware Pruning**:
    *   Auto-detect Method/Conclusion chapters.
    *   Auto-exclude References/Related Work.
    *   Only perform OCR on pages with visual information.
*   **FR-1.4**: Output must be structured JSON + Markdown.

### Level 2: The Synthesis Engine (Interaction)
*   **FR-2.1**: Support "Cross-Paper Comparison" (e.g., "Compare the throughput of Paper A and Paper B based on their tables").
*   **FR-2.2**: Local RAG retrieval using SQLite FTS5 (Keyword) + Embeddings (Semantic).
*   **FR-2.3**: Generate summaries based on OCR-enhanced structured data.

### Level 3: The Insight Engine (Advanced)
*   **FR-3.1**: Detect "Novelty" by comparing new paper metrics against the local database.
*   **FR-3.2**: (Optional) Visualize citation/idea evolution graphs.

## 3. Non-Functional Requirements (Constraints)
*   **Hardware**: Jetson AGX Orin 64GB.
*   **Environment**: JetPack 5.x, Torch 2.0.0 (No Docker preferred).
*   **Performance**:
    *   Parsing Throughput: > 20 papers/min.
    *   GPU Utilization: > 90% during batch processing (No I/O starvation).

## 4. Success Metrics (For Paper)
1.  **Throughput**: Comparison vs. Standard MinerU and Flash-MinerU.
2.  **Ablation Study**: Prove that "Pruning" reduces latency by significant margin while maintaining summarization quality.
3.  **Heterogeneity**: Prove that offloading Layout/Det to DLA reduces GPU memory footprint.