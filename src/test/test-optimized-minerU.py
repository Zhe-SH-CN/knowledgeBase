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
os.environ["MINERU_MODEL_SOURCE"] = "local"
os.environ["MINERU_DEVICE_MODE"] = "cuda:0"
os.environ["MODELSCOPE_LOG_LEVEL"] = "40"
fitz.TOOLS.mupdf_display_errors(False)

from mineru.cli.common import prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)

# ================= 2. 增强版 CPU 预处理 Worker =================
visual_md = ""


def cpu_pre_process_worker(pdf_path):
    try:
        re_ref = re.compile(r"\n#?\s*(?:References|REFERENCES|Bibliography)", re.I)
        re_visual = re.compile(r"\b(Table|Figure|Fig\.)\s+\d+\b", re.I)
        re_caption = re.compile(
            r"(?:Table|Figure|Fig\.)\s+\d+[:\.]?\s+[^\n]*", re.IGNORECASE
        )

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
                if re_ref.search(txt):
                    idx_ref = i

            if len(page.get_images()) > 0 or re_visual.search(txt):
                # print(f"{'len(page.get_images())':<25}: {len(page.get_images())}, {'visual found':<25}: {bool(re_visual.search(txt))}")
                if idx_ref == -1 or i <= idx_ref:
                    ocr_indices.append(i)

        # 1. Front Matter: 前两页全量文本
        front_text = ""
        for i in range(min(2, total_pages)):
            front_text += page_raw_texts.get(i, "") + "\n"

        # all_visual = re_visual.findall(front_text)
        # print(all_visual)

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
            "all_texts_dict": page_raw_texts,  # 传递给保存进程用于切片
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "name": Path(pdf_path).stem}


# ================= 3. CPU 多核保存与切片 Worker (v3.4) =================


def cpu_save_worker(data_pack):
    """
    负责：
    1. 文本精准切片（Conclusion 提取）
    2. 视觉组件渲染
    3. 图片规范化重命名 (pdfname-idx.jpg)
    """
    (middle_json_dict, meta, output_root) = data_pack
    name = meta["name"]
    try:
        # 定义输出路径，不加 pipeline 前缀
        paper_folder = Path(output_root) / name
        img_folder = paper_folder / "images"
        os.makedirs(img_folder, exist_ok=True)

        # --- A. 文本精准切片逻辑 ---
        # 拼合全文文本
        full_text = "\n".join(
            [meta["all_texts_dict"][i] for i in sorted(meta["all_texts_dict"].keys())]
        )

        # 在前两页的front_text中寻找 Abstract 和 Introduction 的位置
        front_text = meta["front_text"]
        re_abst = re.compile(r"(Abstract|ABSTRACT)", re.M)
        re_intro = re.compile(
            r"\n\s*(?:1\.?\s+)?(Introduction|INTRODUCTION)", re.I | re.M
        )

        abs_m = re_abst.search(front_text)
        intro_m = re_intro.search(front_text)

        metadata_part = front_text[: abs_m.start()].strip() if abs_m else "Not Found"

        if abs_m and intro_m:
            abstract_part = front_text[abs_m.start() : intro_m.start()].strip()
            # Introduction 取从标题开始到后续 3000 字符（防止太长）
            introduction_part = front_text[
                intro_m.start() : intro_m.start() + 3000
            ].strip()
        else:
            abstract_part = "Not Found"
            introduction_part = "Not Found"

        # 正则寻找 Conclusion
        re_conc = re.compile(
            r"\n#?\s*(?:\d\.?\s+)?(?:Conclusion|CONCLUSION|Summary)", re.I | re.M
        )
        re_ref = re.compile(
            r"\n#?\s*(?:References|REFERENCES|Bibliography|参考文献)", re.I | re.M
        )
        re_related = re.compile(
            r"\n#?\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK)", re.I | re.M
        )
        re_stop = re.compile(
            r"\n#?\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK|References|REFERENCES|Bibliography|Appendix|APPENDIX|Acknowledgements)",
            re.I,
        )

        conc_final = "Conclusion not identified."
        conc_match = re_conc.search(full_text)
        if conc_match:
            start_pos = conc_match.start()
        # 从 Conclusion 之后寻找终点锚点
        after_conc = full_text[conc_match.end() :]
        stop_match = re_stop.search(after_conc)
        if stop_match:
            conc_final = full_text[start_pos : conc_match.end() + stop_match.start()]
        else:
            conc_final = full_text[
                start_pos : start_pos + 1500
            ]  # 没找到终点则截取 1500 字

        all_pages_text = meta["all_texts_dict"]

        # --- B. Caption匹配与视觉组件处理 ---
        re_caption = re.compile(
            r"(?:Table|Figure|Fig\.)\s+\d+[:\.]?\s+[^\n]*", re.IGNORECASE
        )
        visual_md = ""

        if middle_json_dict:
            # 从minerU的高质量文本中提取Caption
            captions = []
            if "pdf_info" in middle_json_dict:
                pdf_info = middle_json_dict["pdf_info"]
                # 遍历所有页面提取文本
                if isinstance(pdf_info, list):
                    for page_info in pdf_info:
                        if isinstance(page_info, dict) and "texts" in page_info:
                            for text_item in page_info["texts"]:
                                if isinstance(text_item, dict) and "text" in text_item:
                                    text_content = text_item["text"]
                                    found_captions = re_caption.findall(text_content)
                                    captions.extend(found_captions)
            # 执行渲染 (图片会自动存入 img_folder)
            visual_img_md = pipeline_union_make(
                middle_json_dict["pdf_info"], MakeMode.MM_MD, "images"
            )

            # 物理重命名图片：随机哈希 -> 论文名-index
            # 排序确保 index 相对稳定
            cur_imgs = sorted(
                [f for f in os.listdir(img_folder) if f.endswith((".jpg", ".png"))]
            )
            for i, old_name in enumerate(cur_imgs):
                ext = os.path.splitext(old_name)[1]
                new_name = f"{name}-{i}{ext}"
                # 物理重命名
                os.rename(img_folder / old_name, img_folder / new_name)
                # 替换 Markdown 中的路径引用
                visual_img_md = visual_img_md.replace(
                    f"images/{old_name}", f"images/{new_name}"
                )

            # 生成Caption的无序列表
            if captions:
                caption_list_md = "\n".join([f"- {caption}" for caption in captions])
            else:
                caption_list_md = "- No captions found"

            # 组合Caption列表和图片引用
            visual_md = f"""{caption_list_md}

{visual_img_md}"""

            # 保存visual.md文件
            visual_md_content = f"""# {name} Visual Components

## Captions
{caption_list_md}

## Images
{visual_img_md}
"""
            with open(
                paper_folder / f"{name}_visual.md",
                "w",
                encoding="utf-8",
                errors="replace",
            ) as f:
                f.write(visual_md_content)

        # --- C. 缝合最终报告 ---
        final_md = f"""# {name} Analysis Report
## 0. Meta(Before Abstract)
{metadata_part}

## 1. Abstract
{abstract_part}

---
## 2. Introduction
{introduction_part}

## 3. Methodology
Skipped

## 4. Conclusion & Findings
{conc_final}

---
## 5. Visual Components
{visual_md}

*Generated by EdgeScholar Heterogeneous Pipeline v3.6*
"""
        with open(
            paper_folder / f"{name}_report.md", "w", encoding="utf-8", errors="replace"
        ) as f:
            f.write(final_md)

        return True
    except Exception as e:
        logger.error(f"Error saving {name}: {e}")
        return False


# ================= 4. 主执行引擎 =================


class EdgeScholarBatchEngine:
    def __init__(self, output_root):
        self.output_root = output_root

    def run_benchmark(self, pdf_folder, batch_size=10):
        abs_folder = os.path.abspath(pdf_folder)
        pdf_paths = [
            os.path.join(abs_folder, f)
            for f in os.listdir(abs_folder)
            if f.lower().endswith(".pdf")
        ][:batch_size]

        logger.info("🔥 预热显卡资源...")
        sample_path = "./input/sample.pdf"
        if os.path.exists(sample_path):
            _ = pipeline_doc_analyze(
                [open(sample_path, "rb").read()], ["en"], formula_enable=False
            )

        # Step 2: CPU 并行扫描
        t_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(len(pdf_paths), 10)) as executor:
            meta_list = list(executor.map(cpu_pre_process_worker, pdf_paths))
        valid_meta = [m for m in meta_list if m["status"] == "success"]

        # Step 3: GPU 批量推理
        ocr_needed_data = [m for m in valid_meta if m["ocr_bytes"] is not None]
        serializable_results = {}

        if ocr_needed_data:
            logger.info(f"🚀 GPU 推理: {len(ocr_needed_data)} 篇含图论文...")
            batch_bytes = [m["ocr_bytes"] for m in ocr_needed_data]
            results = pipeline_doc_analyze(
                batch_bytes,
                ["en"] * len(batch_bytes),
                formula_enable=False,
                table_enable=False,
            )

            # --- 关键修复：在主进程转为纯 Dict，解决 ctypes 序列化问题 ---
            logger.info("⚡ 转换 C 对象为可序列化 Dict...")
            for i, m in enumerate(ocr_needed_data):
                paper_img_dir = Path(self.output_root) / m["name"] / "images"
                os.makedirs(paper_img_dir, exist_ok=True)

                # 这一步会将 results 里的指针解构成可序列化的字典数据
                # 注意：为了获取完整的 middle_json，必须传入 image_writer
                image_writer = FileBasedDataWriter(str(paper_img_dir))
                middle_json_dict = pipeline_result_to_middle_json(
                    results[0][i],
                    results[1][i],
                    results[2][i],
                    image_writer,
                    "en",
                    True,
                    formula_enabled=False,
                )
                serializable_results[m["name"]] = middle_json_dict

        # Step 4: 多核并行保存
        logger.info("💾 多核并行保存 v3.8 (图片重命名 + 文本深度切片)...")
        save_tasks = []
        for m in valid_meta:
            res_dict = serializable_results.get(m["name"], None)
            save_tasks.append((res_dict, m, self.output_root))

        with ProcessPoolExecutor(max_workers=min(len(save_tasks), 8)) as executor:
            list(executor.map(cpu_save_worker, save_tasks))

        logger.info(
            f"📊 平均耗时: {((time.perf_counter() - t_start) / len(valid_meta)):.2f} seconds/paper"
        )


if __name__ == "__main__":
    engine = EdgeScholarBatchEngine("./output/mineru_batch_v3.8")
    engine.run_benchmark("./input/osdi2025", batch_size=10)
