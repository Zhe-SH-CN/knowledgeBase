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
        re_ref = re.compile(r'\n#?\s*(?:References|REFERENCES|Bibliography)', re.I)
        re_visual = re.compile(r'\b(Table|Figure|Fig\.)\s+\d+\b', re.I)

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        page_raw_texts = {}
        ocr_indices = []
        idx_ref = -1
        
        for i in range(total_pages):
            page = doc[i]
            blocks = page.get_text("blocks", sort=True)
            txt = "\n".join([b[4] for b in blocks if b[6] == 0])
            page_raw_texts[i] = txt
            
            if idx_ref == -1 and i > total_pages * 0.3:
                if re_ref.search(txt): idx_ref = i
            
            if len(page.get_images()) > 0 or re_visual.search(txt):
                if idx_ref == -1 or i <= idx_ref:
                    ocr_indices.append(i)

        # 1. Front Matter: 前两页全量文本
        front_text = ""
        for i in range(min(2, total_pages)):
            front_text += page_raw_texts.get(i, "") + "\n"

        # 2. 生成用于推理的微型 PDF
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
            "all_texts_dict": page_raw_texts, # 传递给保存进程用于切片
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "name": Path(pdf_path).stem}

# ================= 3. CPU 多核保存与切片 Worker (v3.4) =================

def cpu_save_worker(data_pack):
    """
    负责：
    1. 文本精准切片 (Before Abstract / Abstract / Intro)
    2. 视觉组件顺序重命名 (和论文出现顺序一致)
    3. 结论提取 (从 all_texts_dict 动态计算)
    """
    (middle_json_dict, meta, output_root) = data_pack
    name = meta['name']
    
    try:
        paper_folder = Path(output_root) / name
        img_folder = paper_folder / "images"
        
        # --- A. 视觉组件顺序处理 ---
# --- B. 视觉组件处理 (修复路径拼接与递归搜索) ---
        visual_md = ""
        if middle_json_dict:
            img_idx = 0
            # 记录 {旧文件名(hash): 新文件名(index)}，防止多处引用同一张图时重复重命名
            renamed_map = {} 
            
            # 1. 定义递归查找器：找出所有含图片的 block
            def get_visual_blocks(obj):
                found = []
                if isinstance(obj, dict):
                    # 判断是否是视觉块
                    if obj.get("img_path") or obj.get("table_img_path"):
                        found.append(obj)
                    # 递归查找子元素
                    for k, v in obj.items():
                        found.extend(get_visual_blocks(v))
                elif isinstance(obj, list):
                    for item in obj:
                        found.extend(get_visual_blocks(item))
                return found

            # 获取所有视觉块
            all_visual_blocks = get_visual_blocks(middle_json_dict)

            # 2. 处理图片重命名与 MD 生成
            for block in all_visual_blocks:
                # 获取原始路径 (可能是 "images/xxx_hash.jpg" 或 "xxx_hash.jpg")
                raw_rel_path = block.get("img_path") or block.get("table_img_path")
                if not raw_rel_path: continue

                # 【关键修复】：只提取文件名，忽略 JSON 里的目录前缀
                hash_filename = Path(raw_rel_path).name 
                
                # 构造物理路径
                old_file_path = img_folder / hash_filename
                
                # 确定新文件名
                if hash_filename in renamed_map:
                    # 如果已经重命名过（同一张图被多次引用），直接复用
                    final_name = renamed_map[hash_filename]
                else:
                    # 如果是新图，生成新名字
                    if old_file_path.exists():
                        ext = old_file_path.suffix
                        new_name = f"{name}-{img_idx}{ext}"
                        new_file_path = img_folder / new_name
                        
                        try:
                            os.rename(old_file_path, new_file_path)
                            # 记录映射关系
                            renamed_map[hash_filename] = new_name
                            final_name = new_name
                            img_idx += 1
                        except OSError:
                            # 如果重命名失败（极少见），沿用旧名
                            final_name = hash_filename
                    else:
                        # 图片文件物理丢失，跳过生成 MD
                        # logger.warning(f"Image missing: {old_file_path}")
                        continue

                # 3. 生成 Markdown
                # 区分表格和图片
                block_type = block.get("type", "").lower()
                tag = "📊 Table" if "table" in block_type else "🖼️ Figure"
                caption = block.get("caption", "").strip()
                
                visual_md += f"### {tag} {img_idx} (Source: Page {block.get('page_idx', '?')})\n"
                if caption:
                    visual_md += f"> **Caption:** {caption}\n\n"
                
                # 写入图片链接
                visual_md += f"![](images/{final_name})\n\n"
        # --- B. 文本切片逻辑 (包含之前缺失的 conclusion_text 计算) ---
        # 1. 拼合全文用于搜索
        all_pages_indices = sorted(meta['all_texts_dict'].keys())
        full_raw_text = "\n".join([meta['all_texts_dict'][i] for i in all_pages_indices])
        
        # 2. 前部切片 (Metadata/Abstract/Intro)
        abs_regex = re.compile(r'(Abstract|ABSTRACT)', re.M)
        intro_regex = re.compile(r'\n\s*(?:1\.?\s+)?(Introduction|INTRODUCTION)', re.I | re.M)
        
        abs_m = abs_regex.search(full_raw_text)
        intro_m = intro_regex.search(full_raw_text)
        
        metadata_part = full_raw_text[:abs_m.start()].strip() if abs_m else "Not Found"
        
        if abs_m and intro_m:
            abstract_part = full_raw_text[abs_m.start():intro_m.start()].strip()
            # Introduction 取从标题开始到后续 3000 字符（防止太长）
            introduction_part = full_raw_text[intro_m.start():intro_m.start()+4000].strip()
        else:
            abstract_part = "Not Found"
            introduction_part = "Not Found"

        # 3. 结论切片 (从 all_texts_dict 动态提取并剔除 Related Work)
        re_conc = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Conclusion|CONCLUSION|Summary)', re.I | re.M)
        re_ref = re.compile(r'\n#?\s*(?:References|REFERENCES|Bibliography|参考文献)', re.I | re.M)
        re_related = re.compile(r'\n#?\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK)', re.I | re.M)

        conclusion_final = "Conclusion not identified."
        conc_m = re_conc.search(full_raw_text)
        if conc_m:
            # 从结论开始，往后找参考文献
            post_conc_text = full_raw_text[conc_m.start():]
            # 先切掉参考文献
            pre_ref_text = re_ref.split(post_conc_text)[0]
            # 再切掉可能存在的 Related Work (如果它在结论之后)
            clean_conc = re_related.split(pre_ref_text)[0]
            conclusion_final = clean_conc.strip()
        else:
            # 兜底：如果没找到结论标题，取全文最后 1500 字符（剔除参考文献后）
            pre_ref_text = re_ref.split(full_raw_text)[0]
            conclusion_final = pre_ref_text[-1500:].strip()

        # --- C. 缝合最终报告 ---
        final_md = f"""# Paper: {name}

## 1. Metadata (Before Abstract)
{metadata_part}

## 2. Abstract
{abstract_part}

## 3. Introduction
{introduction_part}

## 4. Methodology
(Methodology section is skipped)

## 5. Conclusion & Findings
{conclusion_final}

## 6. Visual Components (In Appearance Order)
{visual_md if visual_md else "No tables or figures found."}

---
*Generated by EdgeScholar Optimized MD v3.8.1*
"""
        with open(paper_folder / f"{name}_report.md", "w", encoding="utf-8", errors="replace") as f:
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
        
        logger.info("🔥 预热显卡资源...")
        sample_path = "./input/sample.pdf"
        if os.path.exists(sample_path):
            _ = pipeline_doc_analyze([open(sample_path, "rb").read()], ['en'], formula_enable=False)

        # Step 2: CPU 并行扫描
        t_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(len(pdf_paths), 10)) as executor:
            meta_list = list(executor.map(cpu_pre_process_worker, pdf_paths))
        valid_meta = [m for m in meta_list if m['status'] == 'success']

        # Step 3: GPU 批量推理
        ocr_needed_data = [m for m in valid_meta if m['ocr_bytes'] is not None]
        serializable_results = {}
        
        if ocr_needed_data:
            logger.info(f"🚀 GPU 推理: {len(ocr_needed_data)} 篇含图论文...")
            batch_bytes = [m['ocr_bytes'] for m in ocr_needed_data]
            results = pipeline_doc_analyze(batch_bytes, ['en']*len(batch_bytes), formula_enable=False, table_enable=False)
            
            # --- 关键修复：在主进程转为纯 Dict，解决 ctypes 序列化问题 ---
            logger.info("⚡ 转换 C 对象为可序列化 Dict...")
            for i, m in enumerate(ocr_needed_data):
                paper_img_dir = Path(self.output_root) / m['name'] / "images"
                os.makedirs(paper_img_dir, exist_ok=True)
                
                # 这一步会将 results 里的指针解构成可序列化的字典数据
                # 注意：为了获取完整的 middle_json，必须传入 image_writer
                image_writer = FileBasedDataWriter(str(paper_img_dir))
                middle_json_dict = pipeline_result_to_middle_json(
                    results[0][i], results[1][i], results[2][i], 
                    image_writer, "en", True, formula_enabled=False
                )
                serializable_results[m['name']] = middle_json_dict

        # Step 4: 多核并行保存
        logger.info("💾 多核并行保存 v3.6 (图片重命名 + 文本深度切片)...")
        save_tasks = []
        for m in valid_meta:
            res_dict = serializable_results.get(m['name'], None)
            save_tasks.append((res_dict, m, self.output_root))

        with ProcessPoolExecutor(max_workers=min(len(save_tasks), 8)) as executor:
            list(executor.map(cpu_save_worker, save_tasks))
            
        logger.info(f"📊 平均耗时: {((time.perf_counter()-t_start)/len(valid_meta)):.2f} seconds/paper")

if __name__ == "__main__":
    # 更新目录名为 v3.6
    engine = EdgeScholarBatchEngine("./output/mineru_batch_v3.6")
    engine.run_benchmark("./input/osdi2025", batch_size=10)