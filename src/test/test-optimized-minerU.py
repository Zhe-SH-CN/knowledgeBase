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

NONE_SIGNAL = "Not Found / Not Resolved"

# ================= 2. CPU 预处理 Worker =================

def cpu_pre_process_worker(pdf_path):
    try:
        # 更加严谨的 OSDI 风格章节正则
        re_method = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Design|Architecture|Implementation|Evaluation|Experiment)', re.I)
        re_ref = re.compile(r'\n#?\s*(?:References|REFERENCES|Bibliography)', re.I)
        re_visual = re.compile(r'\b(Table|Figure|Fig\.)\s+\d+\b', re.I)

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        page_raw_texts = {}
        ocr_indices = []
        idx_ref = -1
        
        for i in range(total_pages):
            page = doc[i]
            # 解决双栏乱序的关键：sort=True
            blocks = page.get_text("blocks", sort=True)
            txt = "\n".join([b[4] for b in blocks if b[6] == 0])
            page_raw_texts[i] = txt
            
            if idx_ref == -1 and i > total_pages * 0.5 and re_ref.search(txt): 
                idx_ref = i
            
            # 只要包含图片或引用，就标记为需要视觉解析
            if len(page.get_images()) > 0 or re_visual.search(txt):
                if idx_ref == -1 or i <= idx_ref:
                    ocr_indices.append(i)

        # 1. Introduction & Abstract: 直接取前两页全量内容
        intro_and_abstract = page_raw_texts.get(0, "") + "\n" + page_raw_texts.get(1, "")

        # 2. Methodology / Experiments: 只有匹配到 OSDI 常用标题才提取
        full_text_for_search = "\n".join(page_raw_texts.values())
        method_text = NONE_SIGNAL
        m_method = re_method.search(full_text_for_search)
        if m_method:
            # 仅截取该章节开始后的 2000 字符
            method_text = full_text_for_search[m_method.start() : m_method.start() + 2000].strip()

        # 3. Conclusion: 精准拦截逻辑
        re_conc = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Conclusion|CONCLUSION|Summary)', re.I)
        re_stop = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK|References|REFERENCES|Bibliography|Appendix|APPENDIX)', re.I)
        conclusion_text = NONE_SIGNAL
        m_conc = re_conc.search(full_text_for_search)
        if m_conc:
            rest_text = full_text_for_search[m_conc.end():]
            m_stop = re_stop.search(rest_text)
            # 如果后面有 Related Work 或 Ref，就在那里停下
            end_pos = m_conc.end() + m_stop.start() if m_stop else m_conc.start() + 2500
            conclusion_text = full_text_for_search[m_conc.start() : end_pos].strip()

        # 生成微型 PDF
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
            "intro_abstract": intro_and_abstract,
            "methodology": method_text,
            "conclusion": conclusion_text,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "name": Path(pdf_path).stem}

# ================= 3. CPU 结果缝合 Worker =================

def cpu_save_worker(data_pack):
    (middle_json_clean, meta, output_root) = data_pack
    name = meta['name']
    try:
        paper_folder = Path(output_root) / name
        img_folder = paper_folder / "images"
        os.makedirs(img_folder, exist_ok=True)
        
        # --- A. 深度递归搜索视觉资产 ---
        visual_assets = []
        
        def find_assets_recursive(obj):
            """递归遍历所有 JSON 节点寻找图片和表格"""
            if isinstance(obj, dict):
                # 如果是视觉节点
                b_type = obj.get("type", "").lower()
                img_path = obj.get("img_path") or obj.get("table_img_path")
                
                if img_path and b_type in ["table", "image", "figure", "table_body"]:
                    visual_assets.append({
                        "type": b_type,
                        "old_name": img_path,
                        "caption": obj.get("caption", "").strip()
                    })
                
                # 继续向下递归
                for value in obj.values():
                    find_assets_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    find_assets_recursive(item)

        if middle_json_clean:
            find_assets_recursive(middle_json_clean)

        # --- B. 物理重命名与链接生成 ---
        final_visual_md = ""
        asset_idx = 0
        
        # 记录已经处理过的旧文件名，防止同一个图片被重命名两次
        processed_old_names = set()

        for asset in visual_assets:
            old_name = asset['old_name']
            if old_name in processed_old_names: continue
            
            old_p = img_folder / old_name
            if old_p.exists():
                ext = old_p.suffix
                new_name = f"{name}-{asset_idx}{ext}"
                new_p = img_folder / new_name
                
                # 执行重命名
                os.rename(old_p, new_p)
                processed_old_names.add(old_name)
                
                # 构造 MD 输出
                tag = "📊 Table" if "table" in asset['type'] else "🖼️ Figure"
                final_visual_md += f"## {tag}: {new_name}\n"
                if asset['caption']:
                    final_visual_md += f"> **Caption:** {asset['caption']}\n\n"
                final_visual_md += f"![](images/{new_name})\n\n"
                asset_idx += 1

        # --- C. 文本切片逻辑 (保持之前的稳定逻辑) ---
        all_pages_list = [meta['all_pages_text'][i] for i in sorted(meta['all_pages_text'].keys())]
        full_text = "\n".join(all_pages_list)
        
        # 1. Methodology / Experiment
        re_method = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Design|Architecture|Implementation|Evaluation|Experiment)', re.I)
        m_match = re_method.search(full_text)
        method_text = full_text[m_match.start():m_match.start()+2500] if m_match else "Not Found"

        # 2. Conclusion (拦截逻辑)
        re_conc = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Conclusion|CONCLUSION|Summary)', re.I)
        re_stop = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK|References|REFERENCES|Bibliography|Appendix|APPENDIX)', re.I)
        conc_text = "Not Found"
        conc_match = re_conc.search(full_text)
        if conc_match:
            rest = full_text[conc_match.end():]
            stop_match = re_stop.search(rest)
            end_pos = conc_match.end() + stop_match.start() if stop_match else conc_match.start() + 2500
            conc_text = full_text[conc_match.start() : end_pos]

        # --- D. 组装最终报告 ---
        final_md = f"""# Paper: {name}

## Abstract & Introduction
{meta['intro_abstract'].strip()}

## Methodology / Experiments
{method_text.strip()}

## Conclusion
{conc_text.strip()}

## Figures & Tables
{final_visual_md if final_visual_md else "No visual assets matched in JSON."}

---
*EdgeScholar Optimized MD v3.7.1*
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
        pdf_paths = sorted([os.path.join(abs_folder, f) for f in os.listdir(abs_folder) if f.lower().endswith(".pdf")])[:batch_size]
        
        logger.info("🔥 启动 v3.7 高吞吐流水线...")
        sample_p = "./input/sample.pdf"
        if os.path.exists(sample_p):
            _ = pipeline_doc_analyze([open(sample_p, "rb").read()], ['en'], formula_enable=False)

        # 1. CPU 预处理 (并行)
        t_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(len(pdf_paths), 10)) as executor:
            meta_list = list(executor.map(cpu_pre_process_worker, pdf_paths))
        valid_meta = [m for m in meta_list if m['status'] == 'success']

        # 2. GPU 推理 + 主进程解构 (解决 Pickle 问题)
        ocr_needed_data = [m for m in valid_meta if m['ocr_bytes'] is not None]
        serializable_results = {}
        
        if ocr_needed_data:
            logger.info(f"🚀 GPU 推理: 处理 {len(ocr_needed_data)} 篇论文的图表页...")
            batch_bytes = [m['ocr_bytes'] for m in ocr_needed_data]
            results = pipeline_doc_analyze(batch_bytes, ['en']*len(batch_bytes), formula_enable=False, table_enable=False)
            
            logger.info("⚡ 结构解构与图片同步存盘...")
            for i, m in enumerate(ocr_needed_data):
                paper_img_dir = Path(self.output_root) / m['name'] / "images"
                os.makedirs(paper_img_dir, exist_ok=True)
                
                # 这一步将 C 指针对象转为纯 Dict，并在磁盘生成原始图片
                image_writer = FileBasedDataWriter(str(paper_img_dir))
                middle_json = pipeline_result_to_middle_json(
                    results[0][i], results[1][i], results[2][i], 
                    image_writer, "en", True, formula_enabled=False
                )
                serializable_results[m['name']] = middle_json

        # 3. 并行保存
        logger.info("💾 多核异步保存 (精准切片 + 图片重命名)...")
        save_tasks = []
        for m in valid_meta:
            res_dict = serializable_results.get(m['name'], None)
            save_tasks.append((res_dict, m, self.output_root))

        with ProcessPoolExecutor(max_workers=min(len(save_tasks), 8)) as executor:
            list(executor.map(cpu_save_worker, save_tasks))
            
        logger.info(f"📊 平均耗时: {((time.perf_counter()-t_start)/len(valid_meta)):.2f} seconds/paper")

if __name__ == "__main__":
    engine = EdgeScholarBatchEngine("./output/mineru_batch_v3.5")
    engine.run_benchmark("./input/osdi2025", batch_size=10)