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

# ================= 2. 增强版 CPU 预处理 Worker =================

def cpu_pre_process_worker(pdf_path):
    try:
        # 定义严格的章节标题正则 (匹配行首)
        re_conc = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Conclusion|CONCLUSION|Summary)', re.I)
        re_stop = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK|References|REFERENCES|Bibliography|Appendix|APPENDIX)', re.I)
        re_visual = re.compile(r'\b(Table|Figure|Fig\.)\s+\d+\b', re.I)

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        page_raw_texts = {}
        ocr_indices = []
        
        for i in range(total_pages):
            page = doc[i]
            blocks = page.get_text("blocks", sort=True)
            txt = "\n".join([b[4] for b in blocks if b[6] == 0])
            page_raw_texts[i] = txt
            
            # 视觉感知判定
            if len(page.get_images()) > 0 or re_visual.search(txt):
                ocr_indices.append(i)

        # 1. Front Matter: 前两页全量
        front_text = ""
        for i in range(min(2, total_pages)):
            front_text += page_raw_texts.get(i, "") + "\n"

        # 2. Conclusion: 精准切片逻辑
        full_text = "\n".join(page_raw_texts.values())
        conclusion_text = "Not Found"
        
        conc_match = re_conc.search(full_text)
        if conc_match:
            start_pos = conc_match.start()
            # 从 Conclusion 开始往后找第一个停止词（Related Work/Ref/Appendix）
            rest_text = full_text[conc_match.end():]
            stop_match = re_stop.search(rest_text)
            if stop_match:
                # 截取两者之间
                conclusion_text = full_text[start_pos : conc_match.end() + stop_match.start()]
            else:
                # 如果没找到停止词，取之后 2500 字
                conclusion_text = full_text[start_pos : start_pos + 3000]

        # 生成微型 PDF (仅含图表页)
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
            "conclusion_text": conclusion_text,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "name": Path(pdf_path).stem}

# ================= 3. CPU 结果保存与图片重命名 Worker =================

def cpu_save_worker(data_pack):
    (middle_json_dict, meta, output_root) = data_pack
    name = meta['name']
    try:
        # prepare_env 会生成没有 "pipeline" 前缀的目录（需要传参数自定义）
        # 这里手动控制路径，不带 pipeline 
        paper_output_dir = Path(output_root) / name
        img_output_dir = paper_output_dir / "images"
        os.makedirs(img_output_dir, exist_ok=True)
        
        image_writer = FileBasedDataWriter(str(img_output_dir))
        
        visual_md = ""
        if middle_json_dict:
            # 1. 转换结果并保存原始图片
            middle_json = pipeline_result_to_middle_json(
                middle_json_dict['res'], middle_json_dict['imgs'], middle_json_dict['doc'], 
                image_writer, "en", True, formula_enabled=False
            )
            # 2. 获取初始 MD
            visual_md = pipeline_union_make(middle_json["pdf_info"], MakeMode.MM_MD, "images")

            # 3. 图片重命名逻辑：随机哈希 -> pdf名字-index
            # 扫描目录下的图片
            img_files = sorted([f for f in os.listdir(img_output_dir) if f.endswith(('.jpg', '.png'))])
            for i, old_name in enumerate(img_files):
                ext = os.path.splitext(old_name)[1]
                new_name = f"{name}-{i}{ext}"
                # 物理重命名
                os.rename(img_output_dir / old_name, img_output_dir / new_name)
                # 替换 MD 中的引用
                visual_md = visual_md.replace(f"images/{old_name}", f"images/{new_name}")

        # 缝合最终报告
        final_md = f"""# {name} Analysis Report

## 📄 [PART 1] Front Matter
{meta['front_text']}

---

## 🔍 [PART 2] Visual Evidence
{visual_md}

---

## 🏁 [PART 3] Conclusion & Findings
{meta['conclusion_text']}

---
*Generated by EdgeScholar Heterogeneous Pipeline v3.4*
"""
        print(len(final_md))
        with open(paper_output_dir / f"{name}_report.md", "w", encoding="utf-8", errors="replace") as f:
            f.write(final_md)
            
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
        
        logger.info("🔥 预热 MinerU...")
        sample_path = "./input/sample.pdf"
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                _ = pipeline_doc_analyze([f.read()], ['en'], formula_enable=False, table_enable=False)

        # Step 2: CPU 并行扫描
        t_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(len(pdf_paths), 10)) as executor:
            meta_list = list(executor.map(cpu_pre_process_worker, pdf_paths))
        valid_meta = [m for m in meta_list if m['status'] == 'success']

        # Step 3: GPU 推理
        ocr_needed_data = [m for m in valid_meta if m['ocr_bytes'] is not None]
        infer_outputs = {}
        
        if ocr_needed_data:
            logger.info(f"🚀 GPU 推理: {len(ocr_needed_data)} 篇含图论文...")
            batch_bytes = [m['ocr_bytes'] for m in ocr_needed_data]
            results = pipeline_doc_analyze(batch_bytes, ['en']*len(batch_bytes), formula_enable=False, table_enable=False)
            
            for i, m in enumerate(ocr_needed_data):
                # 包装为子进程可用的 pack
                infer_outputs[m['name']] = {
                    'res': results[0][i], 
                    'imgs': results[1][i], 
                    'doc': results[2][i]
                }

        # Step 4: 并行保存
        logger.info("💾 多核并行保存 v3.4 (图片重命名 + 精准切片)...")
        save_tasks = []
        for m in valid_meta:
            res_pack = infer_outputs.get(m['name'], None)
            save_tasks.append((res_pack, m, self.output_root))

        with ProcessPoolExecutor(max_workers=min(len(save_tasks), 8)) as executor:
            list(executor.map(cpu_save_worker, save_tasks))
            
        logger.info(f"📊 平均耗时: {((time.perf_counter()-t_start)/len(valid_meta)):.2f} seconds/paper")

if __name__ == "__main__":
    # 输出目录设为 v3.4
    engine = EdgeScholarBatchEngine("./output/mineru_batch_v3.4")
    engine.run_benchmark("./input/osdi2025", batch_size=10)