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
os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
os.environ['MINERU_DEVICE_MODE'] = "cuda:0"
os.environ['MODELSCOPE_LOG_LEVEL'] = '40'

# 静默底层输出
fitz.TOOLS.mupdf_display_errors(False)

from mineru.cli.common import prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

# ================= 2. 剪枝核心逻辑 (由 CPU 进程池执行) =================

def cpu_prune_worker(pdf_path):
    """
    该函数在独立进程中运行，执行 CPU 密集的 PDF 扫描和切片任务
    """
    try:
        re_method = re.compile(r'^\s*(?:2|3|II|III)\.?\s+(?:Method|System|Architecture|Design)', re.I | re.M)
        re_ref = re.compile(r'^\s*(?:References|REFERENCES|Bibliography)', re.I | re.M)
        re_visual = re.compile(r'\b(Table|Figure|Fig\.)\s+\d+\b', re.I)

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        target_pages = {0, 1}
        idx_ref = -1
        page_texts = []

        for i, page in enumerate(doc):
            # 极速提取文本块
            blocks = page.get_text("blocks", sort=True)
            txt = "\n".join([b[4] for b in blocks if b[6] == 0])
            page_texts.append(txt)
            
            # 定位关键章节和图表
            if idx_ref == -1 and i > total_pages * 0.5 and re_ref.search(txt):
                idx_ref = i
            if self_is_visual(page, txt, re_visual):
                if i > 1 and (idx_ref == -1 or i <= idx_ref):
                    target_pages.add(i)

        if idx_ref > 0: target_pages.add(idx_ref - 1)
        
        # 执行切片
        target_indices = sorted(list(target_pages))
        new_doc = fitz.open()
        for p in target_indices:
            new_doc.insert_pdf(doc, from_page=p, to_page=p)
        
        pruned_bytes = new_doc.tobytes(garbage=3, deflate=True)
        doc.close()
        new_doc.close()
        
        return {
            "name": Path(pdf_path).stem,
            "bytes": pruned_bytes,
            "page_texts": page_texts,
            "idx_ref": idx_ref,
            "target_indices": target_indices,
            "total_pages": total_pages
        }
    except Exception as e:
        return {"error": str(e), "name": Path(pdf_path).stem}

def self_is_visual(page, txt, re_visual):
    if len(page.get_images()) > 0: return True
    if re_visual.search(txt): return True
    return False

# ================= 3. 批量处理引擎 =================

class EdgeScholarBatchEngine:
    def __init__(self, output_root):
        self.output_root = output_root

    def run_benchmark(self, pdf_folder, batch_size=10):
        pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")][:batch_size]
        
        # --- Step 1: 预热 (Warm-up) ---
        logger.info("🔥 正在预热模型 (消除加载开销)...")
        first_pdf_bytes = open(pdf_paths[0], "rb").read()
        # 强制加载权重到显存
        _ = pipeline_doc_analyze([first_pdf_bytes], ['en'], formula_enable=False, table_enable=False)
        logger.info("✅ 预热完成。")

        # --- Step 2: CPU 并行剪枝 ---
        logger.info(f"⚙️ 正在并行剪枝 {len(pdf_paths)} 篇论文...")
        t_cpu_start = time.perf_counter()
        
        with ProcessPoolExecutor(max_workers=min(len(pdf_paths), 10)) as executor:
            pruned_data_list = list(executor.map(cpu_prune_worker, pdf_paths))
        
        # 过滤掉失败的
        valid_data = [d for d in pruned_data_list if "error" not in d]
        cpu_duration = time.perf_counter() - t_cpu_start
        logger.info(f"✅ CPU 剪枝耗时: {cpu_duration:.2f}s (平均: {cpu_duration/len(valid_data):.2f}s/篇)")

        # --- Step 3: GPU 批量推理 (核心吞吐量测试) ---
        logger.info(f"🚀 开始 GPU 批量推理 (Batch Size: {len(valid_data)})...")
        t_gpu_start = time.perf_counter()
        
        batch_bytes = [d['bytes'] for d in valid_data]
        # 调用核心分析 API
        infer_results, all_images, all_docs, langs, ocrs = pipeline_doc_analyze(
            batch_bytes, ['en'] * len(valid_data), 
            formula_enable=False, table_enable=True
        )
        
        gpu_duration = time.perf_counter() - t_gpu_start
        logger.info(f"⚡ GPU 批量推理完成！耗时: {gpu_duration:.2f}s (平均: {gpu_duration/len(valid_data):.2f}s/篇)")

        # --- Step 4: 结果保存 ---
        logger.info("💾 正在保存结构化 Markdown 报告...")
        for i, data in enumerate(valid_data):
            self.save_paper_result(data, infer_results[i], all_images[i], all_docs[i], langs[i], ocrs[i])

        total_time = cpu_duration + gpu_duration
        print("\n" + "="*50)
        print(f"📊 批处理性能报告 (n={len(valid_data)})")
        print("-" * 50)
        print(f"平均 CPU 剪枝耗时:   {cpu_duration/len(valid_data):.4f}s")
        print(f"平均 GPU 推理耗时:   {gpu_duration/len(valid_data):.4f}s")
        print(f"单篇平均处理速度:    {total_time/len(valid_data):.4f}s")
        print(f"系统总吞吐量:        {60 / (total_time/len(valid_data)):.2f} papers/min")
        print("="*50)

    def save_paper_result(self, data, res, imgs, doc, lang, ocr_en):
        name = data['name']
        try:
            local_image_dir, local_md_dir = prepare_env(self.output_root, name, "pipeline")
            image_writer = FileBasedDataWriter(local_image_dir)
            
            # 结果转换并保存图片
            middle_json = pipeline_result_to_middle_json(
                res, imgs, doc, image_writer, lang, ocr_en, formula_enabled=False
            )
            
            # 获取视觉占位符
            visual_md = pipeline_union_make(middle_json["pdf_info"], MakeMode.MM_MD, str(Path(local_image_dir).name))
            
            # 保存 Markdown
            report_md = f"# {name}\n\n[Pages Analyzed: {data['target_indices']}]\n\n"
            report_md += "---\n## Visual Evidence\n" + visual_md
            
            # --- 关键修复点：增加 errors='replace' 防止编码崩溃 ---
            save_path = Path(local_md_dir) / f"{name}_report.md"
            with open(save_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(report_md)
                
        except Exception as e:
            logger.error(f"保存结果时出错 {name}: {e}")

if __name__ == "__main__":
    engine = EdgeScholarBatchEngine("./output_batch_test")
    # 一次性跑 10 篇
    engine.run_benchmark("./osdi2025", batch_size=10)