# EdgeScholar：边缘端高通量学术知识构建系统

[English](../README.md) | 简体中文

![状态](https://img.shields.io/badge/状态-研究预览-blue) ![硬件](https://img.shields.io/badge/硬件-Jetson_AGX_Orin-green) ![目标](https://img.shields.io/badge/目标-NeurIPS_2026-red)

**EdgeScholar** 是一个隐私优先、高通量的学术论文解析与知识检索系统，专为资源受限的边缘设备（特别是 NVIDIA Jetson AGX Orin）设计。

与云端解决方案（如 MinerU、GPT-4）优先考虑渲染完美度而牺牲延迟和隐私不同，EdgeScholar 专注于**内容感知剪枝**和**异构计算**，以在保持语义保真度的同时最大化**论文/小时**吞吐量。

## 🚀 核心特性

- **⚡ 超高吞吐量**：针对数千篇论文的离线索引进行优化。实现比完整 OCR 流水线快 10-20 倍的速度。
- **🧠 内容感知剪枝**：自动识别高价值页面（方法论、实验）并跳过冗余内容（参考文献、相关工作）以节省计算资源。
- **⚙️ 异构加速**：
  - **CPU**：文本提取（PyMuPDF）和识别（SVTR-Tiny）。
  - **DLA（深度学习加速器）**：布局分析（DocLayout-YOLO）和文本检测（DBNet++）。
  - **GPU**：语义合成（Qwen2.5-3B）。
- **🔒 隐私优先**：100% 设备端执行。数据不会离开本地网络。

## 🛠️ 技术栈

- **语言**：Python 3.8+
- **硬件**：NVIDIA Jetson AGX Orin（JetPack 5.x）/ RTX 3090（开发环境）
- **模型**：
  - 布局：`DocLayout-YOLO`（ONNX/TensorRT）
  - OCR 检测：`DBNet++`（ONNX/TensorRT）
  - OCR 识别：`SVTR-Tiny`（ONNX Runtime）
  - LLM：`Qwen2.5-3B-Instruct`（Transformers/FP16）
- **核心库**：`PyMuPDF`、`RapidOCR-ONNXRuntime`、`Ultralytics`

## 📊 性能目标

| 方法 | 平均延迟（秒/篇） | 吞吐量（篇/分钟） | 资源消耗 |
| :--- | :---: | :---: | :--- |
| 仅 PyMuPDF | 0.2秒 | ~300 | 仅 CPU |
| MinerU（完整） | ~35.0秒 | ~1.7 | 重 GPU |
| **EdgeScholar（ ours）** | **~2.5秒** | **~24** | **混合（CPU+DLA+GPU）** |

## 📦 安装

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/EdgeScholar.git
cd EdgeScholar

# 2. 安装依赖（推荐使用 uv）
uv pip install pymupdf opencv-python numpy ultralytics rapidocr-onnxruntime

# 3. 模型准备
# 详见 scripts/download_models.sh 中的 ONNX 链接
```

## 📂 项目结构

```
EdgeScholar/
├── src/                    # 源代码
│   └── test/              # 测试模块（更多模块即将添加）
├── models/                # ONNX/TensorRT 模型权重
├── input/                 # 输入的学术论文（git 忽略）
├── output/                # 处理结果和基准测试（git 忽略）
├── docs/                  # 文档
│   ├── README.zh-CN.md   # 简体中文文档
│   ├── ARCHITECTURE.md
│   ├── PRD.md
│   ├── SCHEDULE.md
│   ├── AGENTS.md
│   └── pdf2md.md
├── .gitignore
├── README.md
└── ...
```

## 🗓️ 路线图

- [ ] 第 1 级：内容感知解析和结构化元数据提取。

- [ ] 第 2 级：本地 RAG 与基于 SQLite 的多论文摘要。

- [ ] 第 3 级：思想演进图与新颖性检测。
