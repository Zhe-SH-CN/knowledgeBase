import os
import time
import json
import gc  # 引入垃圾回收
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区 =================
os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
os.environ['MINERU_DEVICE_MODE'] = "cuda:0" 

# 批处理大小：根据显存和内存调整。3090 (24G) 建议设为 5-8
# 设为 5 意味着每次并行处理 5 篇论文，处理完释放内存，再处理下 5 篇
BATCH_SIZE = 5 

from mineru.cli.common import prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

def read_raw_pdf(pdf_path):
    """读取原始二进制数据"""
    try:
        with open(pdf_path, "rb") as f:
            raw_bytes = f.read()
        return pdf_path, raw_bytes
    except Exception as e:
        logger.error(f"读取失败 {pdf_path}: {e}")
        return None, None

def save_result(pack):
    """保存结果逻辑 (修复参数名)"""
    idx, res, imgs, doc, lang, ocr_en, pdf_path, output_dir = pack
    file_name = pdf_path.stem
    
    local_image_dir, local_md_dir = prepare_env(output_dir, file_name, "pipeline")
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)

    # 【修复】这里参数名应为 formula_enabled (过去版本可能是 formula_enable)
    # 如果报错，请尝试检查 mineru 版本，新版通常是 formula_enabled
    middle_json = pipeline_result_to_middle_json(
        res, imgs, doc, image_writer, lang, ocr_en, 
        formula_enabled=False # <--- 已修正
    )

    image_relative_dir = str(Path(local_image_dir).name)
    md_content = pipeline_union_make(middle_json["pdf_info"], MakeMode.MM_MD, image_relative_dir)

    md_file = Path(local_md_dir) / f"{file_name}.md"
    md_writer.write_string(md_file.name, md_content)

def batch_process(input_dir, output_dir):
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("未找到 PDF 文件")
        return

    logger.info(f"📂 发现 {len(pdf_files)} 篇 PDF，准备处理...")

    # 1. 快速读取所有文件进内存 (PDF文件本身不大，可以全部读入)
    t_load_start = time.time()
    valid_pdfs = []
    raw_bytes_list = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(read_raw_pdf, pdf_files)
        for p_path, p_bytes in results:
            if p_bytes:
                valid_pdfs.append(p_path)
                raw_bytes_list.append(p_bytes)
    
    logger.info(f"✅ 读取完成 {len(valid_pdfs)} 篇，耗时: {time.time()-t_load_start:.2f}s")
    if not valid_pdfs: return

    # 2. 模型预热 (Warm-up)
    # 取第一篇单独跑，初始化 CUDA Context
    logger.info("🔥 [Warmup] 正在进行模型预热...")
    t_warmup_start = time.time()
    _ = pipeline_doc_analyze(
        [raw_bytes_list[0]], ['en'], 
        parse_method="auto", formula_enable=False, table_enable=False
    )
    logger.info(f"🔥 预热完成，耗时: {time.time()-t_warmup_start:.2f}s")

    # 3. Mini-Batch 循环推理 (核心优化)
    # 跳过预热的那一篇，处理剩下的
    remaining_pdfs = valid_pdfs[1:]
    remaining_bytes = raw_bytes_list[1:]
    total_remaining = len(remaining_pdfs)

    if total_remaining == 0:
        logger.info("没有更多文件需要处理")
        return

    logger.info(f"🚀 开始分批处理剩余 {total_remaining} 篇 (Batch Size: {BATCH_SIZE})...")
    
    # 循环切片
    for i in range(0, total_remaining, BATCH_SIZE):
        batch_pdfs = remaining_pdfs[i : i + BATCH_SIZE]
        batch_bytes = remaining_bytes[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        logger.info(f"⚡ [Batch {current_batch_num}] 处理 {len(batch_pdfs)} 篇...")
        t_batch_start = time.time()

        try:
            # --- 推理 ---
            results_pack = pipeline_doc_analyze(
                batch_bytes, 
                ['en'] * len(batch_bytes), 
                parse_method="auto", 
                formula_enable=False, 
                table_enable=False
            )
            
            # --- 保存 ---
            # 解包结果
            infer_results, all_images, all_docs, langs, ocrs = results_pack
            
            for idx, pdf_path in enumerate(batch_pdfs):
                pack = (idx, infer_results[idx], all_images[idx], all_docs[idx], langs[idx], ocrs[idx], pdf_path, output_dir)
                save_result(pack)

            batch_time = time.time() - t_batch_start
            logger.info(f"✅ [Batch {current_batch_num}] 完成，耗时: {batch_time:.2f}s (Avg: {batch_time/len(batch_pdfs):.2f}s/篇)")

        except Exception as e:
            logger.error(f"❌ [Batch {current_batch_num}] 处理失败: {e}")
        
        # --- 内存清理 ---
        # 显式删除引用，并强制 GC，防止图片数据在内存堆积
        del batch_bytes
        del results_pack
        gc.collect() 

    logger.info(f"🎉 所有任务全部完成。输出目录: {output_dir}")

if __name__ == "__main__":
    in_dir = "./osdi2025" 
    out_dir = "./mineru_batch_output"
    
    if os.path.exists(in_dir):
        batch_process(in_dir, out_dir)
    else:
        logger.error(f"输入目录不存在: {in_dir}")