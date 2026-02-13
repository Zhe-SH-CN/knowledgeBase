import os
import time
import re
import json
import fitz
import gc
import pickle
from pathlib import Path
from loguru import logger
from concurrent.futures import ProcessPoolExecutor

# ================= 1. 环境配置 =================
os.environ['MINERU_MODEL_SOURCE'] = "local"
os.environ['MINERU_DEVICE_MODE'] = "cuda:0"
os.environ['MODELSCOPE_LOG_LEVEL'] = '40'
fitz.TOOLS.mupdf_display_errors(False)

from mineru.cli.common import prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

# ================= 2. 增强版 CPU 预处理 Worker =================

def cpu_pre_process_worker(pdf_path):
    try:
        # 更加严格的章节定位正则
        # 匹配：References, Bibliography 等
        re_ref = re.compile(r'\n#?\s*(?:References|REFERENCES|Bibliography|参考文献)', re.I)
        # 匹配：Related Work, Background 等
        re_related = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK|Background|Prior Work)', re.I)
        # 匹配：Figure/Table 引用
        re_visual = re.compile(r'\b(Table|Figure|Fig\.)\s+\d+\b', re.I)

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        page_raw_texts = {}
        ocr_indices = []
        idx_ref = -1
        
        for i in range(total_pages):
            page = doc[i]
            # --- 核心：全量开启 blocks+sort=True ---
            blocks = page.get_text("blocks", sort=True)
            txt = "\n".join([b[4] for b in blocks if b[6] == 0])
            page_raw_texts[i] = txt
            
            # 定位 References 所在页
            if idx_ref == -1 and i > total_pages * 0.5:
                if re_ref.search(txt):
                    idx_ref = i
            
            # 视觉感知判定：只要有图片或 Table/Figure 引用，就标记为 OCR 页
            if len(page.get_images()) > 0 or re_visual.search(txt):
                # 排除参考文献之后的页，减少冗余
                if idx_ref == -1 or i <= idx_ref:
                    ocr_indices.append(i)

        # --- 语义提取逻辑 ---
        # 1. Front Matter: 无损保留前两页全部文本
        front_text = ""
        for i in range(min(2, total_pages)):
            front_text += page_raw_texts.get(i, "") + "\n"

        # 2. Conclusion: 动态回溯提取
        # 逻辑：取 Ref 页 + Ref 前一页，然后切掉不需要的部分
        conclusion_raw = ""
        if idx_ref != -1:
            # 获取 Ref 所在页及其前一页
            start_p = max(2, idx_ref - 1) # 避开前两页
            for p in range(start_p, idx_ref + 1):
                conclusion_raw += page_raw_texts.get(p, "") + "\n"
            
            # 剪枝 A: 切掉 References 及其之后的所有内容
            conclusion_clean = re_ref.split(conclusion_raw)[0]
            # 剪枝 B: 切掉 Related Work (如果在结论后面)
            conclusion_clean = re_related.split(conclusion_clean)[0]
            # 保留该区域最后的 3500 字符（通常涵盖了完整的 Conclusion 和部分 Evaluation 总结）
            conclusion_final = conclusion_clean[-3500:]
        else:
            # 没找到 Ref，取最后两页
            conclusion_final = "\n".join([page_raw_texts.get(i, "") for i in range(max(0, total_pages-2), total_pages)])

        # 生成用于 MinerU 处理的微型 PDF (仅含图表页)
        pruned_bytes = None
        if ocr_indices:
            new_doc = fitz.open()
            for p in ocr_indices:
                new_doc.insert_pdf(doc, from_page=p, to_page=p)        
            pruned_bytes = new_doc.tobytes(garbage=3, deflate=True)
            new_doc.close()
        
        doc.close()
        return {
            "name": Path(pdf_path).stem,
            "ocr_bytes": pruned_bytes,
            "ocr_mapping": ocr_indices,
            "front_text": front_text,
            "conclusion_text": conclusion_final,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "name": Path(pdf_path).stem}

# ================= 3. CPU 结果保存与缝合 Worker =================

def cpu_save_worker(data_pack):
    """
    多核并行处理：保存为精美的 Markdown 文件
    """
    (middle_json_dict, meta, output_root) = data_pack
    name = meta['name']
    try:
        local_image_dir, local_md_dir = prepare_env(output_root, name, "pipeline")
        image_writer = FileBasedDataWriter(local_image_dir)
        
        # 渲染 MinerU 的视觉输出 (表格、公式、图片占位符)
        visual_md = ""
        if middle_json_dict:
            visual_md = pipeline_union_make(middle_json_dict["pdf_info"], MakeMode.MM_MD, str(Path(local_image_dir).name))

        # 构造 Markdown 报告
        report_md = f"""# {name} Analysis Report

## 📄 [PART 1] Front Matter (Abstract & Intro)
{meta['front_text']}

---

## 🔍 [PART 2] Visual Evidence (Tables & Figures)
> **Note:** These assets are extracted via MinerU OCR from relevant pages.
{visual_md}

---

## 🏁 [PART 3] Conclusion & Findings
{meta['conclusion_text']}

---
*Generated by EdgeScholar Heterogeneous Pipeline*
"""
        # 保存 Markdown 文件
        save_path = Path(local_md_dir) / f"{name}_report.md"
        with open(save_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(report_md)
            
        return True
    except Exception as e:
        logger.error(f"Save error for {name}: {e}")
        return False

# ================= 4. 主执行引擎 =================

class EdgeScholarBatchEngine:
    def __init__(self, output_root):
        self.output_root = output_root

    def run_benchmark(self, pdf_folder, batch_size=10):
        abs_folder = os.path.abspath(pdf_folder)
        pdf_paths = [os.path.join(abs_folder, f) for f in os.listdir(abs_folder) if f.lower().endswith(".pdf")][:batch_size]
        
        logger.info("🔥 加载权重并预热...")
        sample_path = "./input/sample.pdf"
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                _ = pipeline_doc_analyze([f.read()], ['en'], formula_enable=False, table_enable=False)

        # Step 2: 多核扫描与精准剪枝
        logger.info(f"⚙️ 正在执行结构化扫描 (CPU 并行)...")
        t_cpu_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(len(pdf_paths), 10)) as executor:
            meta_list = list(executor.map(cpu_pre_process_worker, pdf_paths))
        valid_meta = [m for m in meta_list if m['status'] == 'success']

        # Step 3: GPU 批量推理
        ocr_needed_data = [m for m in valid_meta if m['ocr_bytes'] is not None]
        serialized_outputs = {}
        
        if ocr_needed_data:
            logger.info(f"🚀 启动 GPU 推理，处理 {len(ocr_needed_data)} 篇论文的图表页...")
            batch_bytes = [m['ocr_bytes'] for m in ocr_needed_data]
            results = pipeline_doc_analyze(batch_bytes, ['en']*len(batch_bytes), formula_enable=False, table_enable=False)
            
            # 主进程进行 dict 转换以规避 Pickle 报错
            for i, m in enumerate(ocr_needed_data):
                local_image_dir, _ = prepare_env(self.output_root, m['name'], "pipeline")
                image_writer = FileBasedDataWriter(local_image_dir)
                middle_json = pipeline_result_to_middle_json(
                    results[0][i], results[1][i], results[2][i], 
                    image_writer, "en", True, formula_enabled=False
                )
                serialized_outputs[m['name']] = middle_json

        # Step 4: 多核结果缝合
        logger.info("💾 正在并行生成 Markdown 报告...")
        save_tasks = []
        for m in valid_meta:
            m_dict = serialized_outputs.get(m['name'], None)
            save_tasks.append((m_dict, m, self.output_root))

        with ProcessPoolExecutor(max_workers=min(len(save_tasks), 8)) as executor:
            list(executor.map(cpu_save_worker, save_tasks))
            
        total_dur = time.perf_counter() - t_cpu_start
        logger.info(f"🎉 全部处理完成！输出文件夹: {self.output_root}")
        logger.info(f"📊 系统总吞吐量: {(total_dur/len(valid_meta)):.2f} seconds/paper")

if __name__ == "__main__":
    engine = EdgeScholarBatchEngine("./output/mineru_final_v5")
    engine.run_benchmark("./input/osdi2025", batch_size=10)