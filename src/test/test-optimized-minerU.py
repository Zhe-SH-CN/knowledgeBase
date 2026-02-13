import os
import time
import re
import json
import fitz
import gc
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

# ================= 2. CPU 预处理 Worker (文本提取与剪枝) =================

def cpu_pre_process_worker(pdf_path):
    try:
        re_ref = re.compile(r'\n#?\s*(?:References|REFERENCES|Bibliography)', re.I)
        re_visual = re.compile(r'\b(Table|Figure|Fig\.)\s+\d+\b', re.I)

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        page_raw_texts = {}
        ocr_indices = []
        idx_ref = -1
        
        for i in range(total_pages):
            page = doc[i]
            # 使用 blocks 和 sort=True 解决双栏乱序
            blocks = page.get_text("blocks", sort=True)
            txt = "\n".join([b[4] for b in blocks if b[6] == 0])
            page_raw_texts[i] = txt
            
            if idx_ref == -1 and i > total_pages * 0.5:
                if re_ref.search(txt): idx_ref = i
            
            # 视觉感知：只要有图片或引用就标记 OCR
            if len(page.get_images()) > 0 or re_visual.search(txt):
                if idx_ref == -1 or i <= idx_ref:
                    ocr_indices.append(i)

        # 提取前两页作为 Front Matter
        front_text = ""
        for i in range(min(2, total_pages)):
            front_text += page_raw_texts.get(i, "") + "\n"

        # 生成用于推理的微型 PDF
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
            "all_pages_text": page_raw_texts,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "name": Path(pdf_path).stem}

# ================= 3. CPU 结果缝合 Worker (v3.5 修复版) =================

def cpu_save_worker(data_pack):
    """
    负责：文本切片、图片重命名、Markdown 合成
    """
    (middle_json_clean, meta, output_root) = data_pack
    name = meta['name']
    try:
        paper_folder = Path(output_root) / name
        img_folder = paper_folder / "images"
        os.makedirs(img_folder, exist_ok=True)
        
        # --- A. 文本精准切片 ---
        all_txt = "\n".join([meta['all_pages_text'][i] for i in sorted(meta['all_pages_text'].keys())])
        
        # 1. Methodology / Experiments (截取 2k)
        re_method = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Method|Proposed|System|Architecture|Design|Experiment|Evaluation)', re.I)
        method_match = re_method.search(all_txt)
        method_txt = all_txt[method_match.start():method_match.start()+2000] if method_match else "Methodology not found."

        # 2. Conclusion (定位起点，拦截 Related Work/Ref/Appendix)
        re_conc = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Conclusion|CONCLUSION|Summary)', re.I)
        re_stop = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK|References|REFERENCES|Bibliography|Appendix|APPENDIX)', re.I)
        conc_txt = "Conclusion not found."
        conc_match = re_conc.search(all_txt)
        if conc_match:
            rest = all_txt[conc_match.end():]
            stop_match = re_stop.search(rest)
            # 修复：不直接在 f-string 中进行 join 操作
            end_idx = conc_match.end() + stop_match.start() if stop_match else conc_match.start() + 2500
            conc_txt = all_txt[conc_match.start() : end_idx]

        # --- B. 视觉元素解析与图片重命名 ---
        visual_md_list = []
        if middle_json_clean:
            asset_idx = 0
            for page in middle_json_clean.get("pdf_info", []):
                for block in page.get("pre_markdown_res", []):
                    if block.get("type") in ["table", "image", "figure"]:
                        raw_img_path = block.get("img_path") or block.get("table_img_path")
                        if not raw_img_path: continue
                        
                        old_p = img_folder / raw_img_path
                        if old_p.exists():
                            ext = old_p.suffix
                            new_name = f"{name}-{asset_idx}{ext}"
                            os.rename(old_p, img_folder / new_name)
                            
                            # 构造 MD 片段
                            cap = block.get("caption", "").strip()
                            tag = "📊 Table" if block["type"] == "table" else "🖼️ Figure"
                            item_md = f"## {tag}: {new_name}\n"
                            if cap: item_md += f"> **Caption:** {cap}\n\n"
                            item_md += f"![](images/{new_name})\n"
                            visual_md_list.append(item_md)
                            asset_idx += 1

        # --- C. 组装 Markdown ---
        # 修复 f-string backslash error: 先在外部 join
        visual_section = "\n".join(visual_md_list) if visual_md_list else "No visual assets extracted."
        
        final_md = f"""# Paper: {name}

## Abstract & Introduction
{meta['front_text'].strip()}

## Methodology / Experiments
{method_txt.strip()}

## Conclusion
{conc_txt.strip()}

## Figures & Tables
{visual_section}

---
*EdgeScholar Optimized MD v3.5*
"""
        with open(paper_folder / f"{name}_report.md", "w", encoding="utf-8", errors="replace") as f:
            f.write(final_md)
        return True
    except Exception as e:
        logger.error(f"Save error for {name}: {e}")
        return False

# ================= 4. 主引擎 =================

class EdgeScholarBatchEngine:
    def __init__(self, output_root):
        self.output_root = output_root

    def run_benchmark(self, pdf_folder, batch_size=10):
        abs_folder = os.path.abspath(pdf_folder)
        pdf_paths = [os.path.join(abs_folder, f) for f in os.listdir(abs_folder) if f.lower().endswith(".pdf")][:batch_size]
        
        t_all_start = time.perf_counter()

        # 1. Warm-up
        logger.info("🔥 预热模型环境...")
        sample_p = "./input/sample.pdf"
        if os.path.exists(sample_p):
            _ = pipeline_doc_analyze([open(sample_p, "rb").read()], ['en'], formula_enable=False)

        # 2. CPU 预处理 (Scan & Prune)
        t_scan_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(len(pdf_paths), 10)) as executor:
            meta_list = list(executor.map(cpu_pre_process_worker, pdf_paths))
        valid_meta = [m for m in meta_list if m['status'] == 'success']
        scan_dur = time.perf_counter() - t_scan_start
        logger.info(f"✅ CPU 扫描完成，平均每篇耗时: {scan_dur/len(valid_meta):.2f}s")

        # 3. GPU 推理 + 结构解构 (主进程处理以避开 Pickle 错误)
        ocr_needed_data = [m for m in valid_meta if m['ocr_bytes'] is not None]
        serializable_results = {}
        
        t_gpu_start = time.perf_counter()
        if ocr_needed_data:
            logger.info(f"🚀 GPU 推理: 处理 {len(ocr_needed_data)} 篇含图论文...")
            batch_bytes = [m['ocr_bytes'] for m in ocr_needed_data]
            results = pipeline_doc_analyze(batch_bytes, ['en']*len(batch_bytes), formula_enable=False, table_enable=False)
            
            # --- 关键：在主进程转为纯 Dict ---
            for i, m in enumerate(ocr_needed_data):
                # 创建对应论文的图片目录
                paper_img_dir = Path(self.output_root) / m['name'] / "images"
                os.makedirs(paper_img_dir, exist_ok=True)
                
                # 转换为可序列化字典 (此过程会保存原始图片)
                image_writer = FileBasedDataWriter(str(paper_img_dir))
                middle_json = pipeline_result_to_middle_json(
                    results[0][i], results[1][i], results[2][i], 
                    image_writer, "en", True, formula_enabled=False
                )
                serializable_results[m['name']] = middle_json

        gpu_dur = time.perf_counter() - t_gpu_start
        logger.info(f"⚡ GPU 推理与数据脱敏完成，平均每篇: {gpu_dur/len(valid_meta):.2f}s")

        # 4. CPU 多核保存 (Markdown 渲染与重命名)
        t_save_start = time.perf_counter()
        save_tasks = []
        for m in valid_meta:
            clean_dict = serializable_results.get(m['name'], None)
            save_tasks.append((clean_dict, m, self.output_root))

        with ProcessPoolExecutor(max_workers=min(len(save_tasks), 8)) as executor:
            list(executor.map(cpu_save_worker, save_tasks))
        
        save_dur = time.perf_counter() - t_save_start
        logger.info(f"✅ 多核保存完成，平均每篇: {save_dur/len(valid_meta):.2f}s")
            
        logger.info(f"📊 平均耗时: {((time.perf_counter()-t_all_start)/len(valid_meta)):.2f} seconds/paper")

if __name__ == "__main__":
    # 指定输出目录为 v3.5
    engine = EdgeScholarBatchEngine("./output/mineru_batch_v3.5")
    engine.run_benchmark("./input/osdi2025", batch_size=10)