# PDF 解析（PDF 转 Markdown）实现方法笔记

## 1. 目前主要在做
PDF 解析（PDF 转 Markdown）

## 2. 实验进展

- [x] **实验一：PyMuPDF 纯文本提取 + 网络爬虫**
    - **方法**：PyMuPDF 提取 PDF 文本层，正则匹配 DOI/arXiv ID，通过 CrossRef/arXiv API 爬取元数据。
    - **耗时**：文本解析（约 0.2s），网络 I/O 耗时较长（约 4.5s）。
    - **缺点**：读不懂表格、图片语义。

- [x] **实验二：所有页面的全量 MinerU OCR识别 (MinerU的Pipeline 模式)**
    - **方法**：使用 MinerU 的标准 Pipeline，对 PDF 所有页面进行布局分析、OCR 识别及 Markdown 重组。
    - **耗时**：约 15s / 篇。
    - **特点**：解析精度高，但是对于纯文本页面完全可以用pymupdf快速解析

- [x] **实验三：内容感知剪枝 MinerU**
    - **模型链接**：MinerU2.5-2509-1.2B https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
    - **方法**：在 MinerU 基础上引入预处理剪枝逻辑，仅对关键页面进行 OCR。
    - **耗时**：约 10s / 篇（随剪枝率波动）。
    - **剪枝逻辑流程**：
        1.  **CPU 极速扫描**：利用 PyMuPDF 在 CPU 上遍历 PDF 全文。
        2.  **需要OCR页面判定**：
            *   **结构**：保留前两页页（Metadata/Intro）以及参考文献前一页（Conclusion）。
            *   **内容**：检测包含 `Table`、`Figure` 关键词或包含大尺寸图片对象的页面。
        3.  **物理切片**：利用 PyMuPDF 将判定为“需要视觉解析”的页面提取出来，在内存中重组为一个微型 PDF 字节流。
        4.  **定向 OCR**：仅将这个微型 PDF 喂给 MinerU 进行 GPU 推理。
        5.  **结果缝合**：将 MinerU 返回的结构化数据（Markdown 表格/图片）插回 PyMuPDF 提取的纯文本流中，形成完整 Markdown。

- [ ] **实验四：剪枝 Flash-MinerU**
    - **目标**：测试 Flash-MinerU 结合剪枝策略后的极限吞吐量。
    - **预期**：能够更快一点。
    - **担心**：Flash-MinerU仅仅出来一周，我担心对于arm64的支持不够

- [ ] **实验五：DLA异构加速方案 (DocLayout-YOLO + RapidOCR)**
    - **目标**：拆解 MinerU 的 Pipeline，把minerU它用的几个模型扔到dla上
    - **方法**：
        *   **Layout**：使用 DocLayout-YOLO 进行版面分析（计划迁移至 DLA）。
        *   **OCR**：使用 RapidOCR (DBNet + SVTR) 进行文字检测与识别（计划迁移至 DLA 或 CPU 多核并行）。
## 3. 输出
- **纯文字**： 保留前两页页（Metadata/Intro）以及参考文献前一页（Conclusion）。
- **官方metadata**: 通过url, arxiv, doi爬取的数据
- **图片与表格**：表格和图片都只抠图抠出来。
- **公式**： 舍弃，开启公式识别的花销太大。

## 4. 未来工作规划

### 短期计划（1 week）
1.  **完成剩余对照实验**：跑通实验四（Flash-MinerU）和实验五（异构拆分版）。
2.  **Jetson AGX 部署**：
    *   **首选方案**：继续推进 DLA 算子适配，尝试将 DocLayout-YOLO 和 DBNet 编译为 TensorRT Engine 运行在 DLA 上。
    *   **备案方案（Fallback）**：直接在 AGX 上部署“剪枝 MinerU”。
        *   **依据**：在 RTX 3090 上的测试数据显示，去除公式识别后的 MinerU 显存占用仅为 680MB 左右。
        *   **可行性**：AGX Orin 拥有 64GB 统一内存，完全有能力同时运行 Qwen2.5-3B（约 6GB）和 MinerU 解析服务，虽然速度不及 DLA 方案，但能保证系统功能闭环。