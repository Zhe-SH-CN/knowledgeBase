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

# ================= 2. 预处理 Worker (保持你的逻辑) =================

def cpu_pre_process_worker(pdf_path):
    try:
        re_ref = re.compile(r'^\s*(?:References|REFERENCES|Bibliography)', re.I | re.M)
        re_visual = re.compile(r'\b(Table|Figure|Fig\.)\s+\d+\b', re.I)

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        idx_ref = -1
        page_info = [] 
        ocr_indices = []
        
        for i, page in enumerate(doc):
            blocks = page.get_text("blocks", sort=True)
            txt = "\n".join([b[4] for b in blocks if b[6] == 0])
            
            if idx_ref == -1 and i > total_pages * 0.5 and re_ref.search(txt):
                idx_ref = i
            
            if self_is_visual(page, txt, re_visual):
                # 遵循你的逻辑：仅在引用页之前进行 OCR
                if idx_ref == -1 or i <= idx_ref:
                    page_info.append({"type": "ocr", "page_idx": i})
                    ocr_indices.append(i)
            else:
                page_info.append({"type": "text", "content": txt, "page_idx": i})

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
            "page_structure": page_info,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "name": Path(pdf_path).stem}

def self_is_visual(page, txt, re_visual):
    if len(page.get_images()) > 0: return True
    if re_visual.search(txt): return True
    return False

# ================= 3. 多核保存 Worker (接收序列化 dict) =================

def cpu_save_worker(data_pack):
    """
    负责：1. 文本精准切片 2. 缝合视觉组件 3. 异步存盘
    """
    (middle_json_dict, meta, output_root) = data_pack
    name = meta['name']
    try:
        # 准备路径
        local_image_dir, local_md_dir = prepare_env(output_root, name, "pipeline")
        
        # --- 1. 文本切片逻辑 (只取精华) ---
        # 拼合所有文本页
        full_raw_text = ""
        for page in meta['page_structure']:
            if page['type'] == 'text':
                full_raw_text += page['content'] + "\n"
        
        # A. 前两页精华
        front_text = ""
        count = 0
        for p in meta['page_structure']:
            if p['type'] == 'text':
                front_text += p['content'] + "\n"
                count += 1
            if count >= 2: break
        
        # B. 方法论定位 (2000字)
        re_method = re.compile(r'^\s*(?:2|3|II|III)\.?\s+(?:Method|Proposed|System|Architecture|Design)', re.I | re.M)
        m_match = re_method.search(full_raw_text)
        method_text = full_raw_text[m_match.start():m_match.start()+2000] if m_match else ""

        # C. 结论定位 (References 之前 2000字)
        re_ref = re.compile(r'\n#+\s+(?:References|REFERENCES|Bibliography|参考文献)', re.I)
        ref_parts = re_ref.split(full_raw_text)
        pre_ref_text = ref_parts[0]
        # 剔除 Related Work
        re_related = re.compile(r'\n#+\s+(?:Related Work|RELATED WORK)', re.I)
        conclusion_text = re_related.split(pre_ref_text)[0][-2000:]

        # --- 2. 视觉组件解析 (利用序列化后的 dict) ---
        visual_md = ""
        if middle_json_dict:
            # 渲染视觉部分的 Markdown (仅含图片/表格占位符)
            visual_md = pipeline_union_make(middle_json_dict["pdf_info"], MakeMode.MM_MD, str(Path(local_image_dir).name))

        # --- 3. 缝合最终输出 (用于喂给 Qwen) ---
        qwen_prompt = f"# {name}\n\n[FRONT MATTER]\n{front_text[:3000]}\n\n"
        if method_text:
            qwen_prompt += f"[METHODOLOGY SNIPPET]\n{method_text}\n\n"
        qwen_prompt += f"[CONCLUSION SNIPPET]\n{conclusion_text}\n\n"
        qwen_prompt += f"[VISUAL ASSETS]\n{visual_md}"

        # 存盘
        with open(Path(local_md_dir) / f"{name}_qwen_input.txt", "w", encoding="utf-8", errors="replace") as f:
            f.write(qwen_prompt)
            
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
        
        # 1. Warm-up (使用你的 sample.pdf 逻辑)
        logger.info("🔥 正在预热模型...")
        sample_path = "./input/sample.pdf"
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                _ = pipeline_doc_analyze([f.read()], ['en'], formula_enable=False, table_enable=False)
        else:
            logger.warning("预热文件 sample.pdf 不存在，使用第一篇论文预热")
            _ = pipeline_doc_analyze([open(pdf_paths[0], "rb").read()], ['en'], formula_enable=False, table_enable=False)

        # 2. CPU 多核剪枝扫描
        logger.info(f"⚙️ 正在并行扫描 {len(pdf_paths)} 篇论文...")
        t_cpu_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(len(pdf_paths), 10)) as executor:
            meta_list = list(executor.map(cpu_pre_process_worker, pdf_paths))
        valid_meta = [m for m in meta_list if m['status'] == 'success']

        # 3. GPU 批量推理
        ocr_needed_data = [m for m in valid_meta if m['ocr_bytes'] is not None]
        serialized_outputs = {} # 存放脱敏后的字典
        
        if ocr_needed_data:
            logger.info(f"🚀 启动 GPU 推理，处理 {len(ocr_needed_data)} 篇含图页...")
            batch_bytes = [m['ocr_bytes'] for m in ocr_needed_data]
            results = pipeline_doc_analyze(batch_bytes, ['en']*len(batch_bytes), formula_enable=False, table_enable=False)
            
            # --- 关键：在主进程完成 dict 转换，解决 Pickle 报错 ---
            logger.info("⚡ 转换数据结构为可序列化 Dict...")
            for i, m in enumerate(ocr_needed_data):
                # 这一步将 C 对象的 infer_result 转换为纯 Python Dict
                local_image_dir, _ = prepare_env(self.output_root, m['name'], "pipeline")
                image_writer = FileBasedDataWriter(local_image_dir)
                
                # 转换为 dict (注意：这里会产生 I/O 保存图片)
                middle_json = pipeline_result_to_middle_json(
                    results[0][i], results[1][i], results[2][i], 
                    image_writer, "en", True, formula_enabled=False
                )
                serialized_outputs[m['name']] = middle_json

        gpu_time = time.perf_counter() - t_cpu_start
        logger.info(f"⚡ 推理与结构转换完成，耗时: {gpu_time:.2f}s")

        # 4. CPU 多核并行切片与保存 (Markdown生成)
        logger.info("💾 启动多核并行文本切片与保存...")
        t_save_start = time.perf_counter()
        save_tasks = []
        for m in valid_meta:
            m_dict = serialized_outputs.get(m['name'], None)
            save_tasks.append((m_dict, m, self.output_root))

        with ProcessPoolExecutor(max_workers=min(len(save_tasks), 8)) as executor:
            list(executor.map(cpu_save_worker, save_tasks))
            
        save_time = time.perf_counter() - t_save_start
        logger.info(f"✅ 保存耗时: {save_time:.2f}s")
        
        total_dur = time.perf_counter() - t_cpu_start
        logger.info(f"📊 系统总吞吐量: {60 / (total_dur/len(valid_meta)):.2f} papers/min")

if __name__ == "__main__":
    engine = EdgeScholarBatchEngine("./output/mineru_final_v4")
    engine.run_benchmark("./input/osdi2025", batch_size=10)