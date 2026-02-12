import os
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import fitz
import json
import time
import re
import numpy as np
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

# ================= 1. 模型补丁 =================
def patch_model(model):
    m = model.model
    if not hasattr(m, 'set_head_attr'): m.set_head_attr = lambda *args, **kwargs: None
    orig_forward = m.forward
    def new_forward(*args, **kwargs):
        res = orig_forward(*args, **kwargs)
        return res['one2one'] if isinstance(res, dict) else res
    m.forward = new_forward
    m.end2end = True
    return model

# ================= 2. 核心引擎 =================
class OptimizedAcademicEngine:
    def __init__(self, model_path):
        t_init = time.perf_counter()
        raw_model = YOLO(model_path)
        self.layout_model = patch_model(raw_model).to("cuda")
        self.ocr_engine = RapidOCR()
        self.init_time = time.perf_counter() - t_init
        print(f"✅ 引擎初始化成功 | 耗时: {self.init_time:.2f}s")

    def get_clean_sorted_text(self, page):
        blocks = page.get_text("blocks", sort=True)
        text_content = []
        for b in blocks:
            if b[6] == 0:
                text = b[4].strip()
                if text: text_content.append(text)
        return "\n".join(text_content)

    def table_to_md(self, ocr_results):
        if not ocr_results: return ""
        ocr_results.sort(key=lambda x: x[0][0][1])
        rows = []
        if ocr_results:
            curr_row, last_y = [ocr_results[0]], ocr_results[0][0][0][1]
            for i in range(1, len(ocr_results)):
                if abs(ocr_results[i][0][0][1] - last_y) < 18:
                    curr_row.append(ocr_results[i])
                else:
                    curr_row.sort(key=lambda x: x[0][0][0])
                    rows.append(curr_row)
                    curr_row, last_y = [ocr_results[i]], ocr_results[i][0][0][1]
            curr_row.sort(key=lambda x: x[0][0][0])
            rows.append(curr_row)
        md = ""
        for idx, r in enumerate(rows):
            cells = [item[1].replace('|', '\\|').replace('\n', ' ') for item in r]
            md += "| " + " | ".join(cells) + " |\n"
            if idx == 0: md += "| " + " | ".join(["---"] * len(cells)) + " |\n"
        return md

    def process_pdf(self, pdf_path, output_dir="./output_results"):
        perf = {}
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_folder = os.path.join(output_dir, base_name)
        os.makedirs(out_folder, exist_ok=True)

        doc = fitz.open(pdf_path)
        
        # --- Step 1: 文本扫描与结构定位 (CPU) ---
        t1 = time.perf_counter()
        
        page_texts = []
        # 定义正则
        re_method = re.compile(r'^\s*(?:2|3|II|III)\.?\s+(?:Method|Proposed|System|Architecture|Design|Implementation)', re.I | re.M)
        re_conclusion = re.compile(r'^\s*(?:\d\.?\s+)?(?:Conclusion|CONCLUSION|Summary)', re.I | re.M)
        re_related = re.compile(r'^\s*(?:\d\.?\s+)?(?:Related Work|RELATED WORK|Background|Prior Work)', re.I | re.M)
        re_ref = re.compile(r'^\s*(?:References|REFERENCES|Bibliography|参考文献)', re.I | re.M)
        
        # 新增：图表关键词正则 (用于发现潜在的表格页)
        re_visual_hint = re.compile(r'^(?:Table|Figure|Fig\.)\s+\d+', re.I | re.M)

        idx_method = -1
        idx_conc = -1
        idx_ref = -1
        
        # 记录包含 "Table X" 或 "Figure X" 字眼的候选页
        visual_candidate_pages = set()

        for i, page in enumerate(doc):
            # 获取文本
            txt = self.get_clean_sorted_text(page)
            page_texts.append(txt)

            # 结构定位
            if idx_method == -1 and i < len(doc)*0.7 and re_method.search(txt): idx_method = i
            if idx_conc == -1 and i > len(doc)*0.5 and re_conclusion.search(txt): idx_conc = i
            if idx_ref == -1 and re_ref.search(txt): idx_ref = i
            
            # 【优化点】视觉内容启发式扫描
            # 如果文本中包含 "Table 1" 或 PyMuPDF 发现有图片对象，则标记该页
            if re_visual_hint.search(txt) or len(page.get_images()) > 0:
                visual_candidate_pages.add(i)

        if idx_ref == -1: idx_ref = len(doc) - 1
        
        perf['1_text_scan'] = time.perf_counter() - t1

        # --- Step 2: 确定最终 OCR 目标页 ---
        # 逻辑：关键章节页 + 启发式扫描发现的页
        target_pages = visual_candidate_pages.copy()
        
        if idx_method != -1: target_pages.add(idx_method)
        if idx_conc != -1: target_pages.add(idx_conc)
        
        # 始终排除前两页 (Title/Abstract) 和 参考文献页
        target_pages.discard(0)
        target_pages.discard(1)
        if idx_ref != -1:
            for r in range(idx_ref, len(doc)):
                target_pages.discard(r)

        # 限制 OCR 页数上限 (例如最多扫 6 页)，防止某些论文每页都有图导致超时
        # 优先保留后半部分的页面（通常实验结果在后面）
        final_pages_list = sorted(list(target_pages))
        if idx_method!=-1 and idx_method in final_pages_list: final_pages_list.remove(idx_method)
        if idx_conc!=-1 and idx_conc in final_pages_list: final_pages_list.remove(idx_conc)
        if len(final_pages_list) > 6:
            final_pages_list = final_pages_list[-6:]

        print(f"  -> 视觉扫描页码: {final_pages_list}")

        # --- Step 3: 视觉增强解析 (DLA/GPU) ---
        t2 = time.perf_counter()
        visual_metadata = []
        
        for p_idx in final_pages_list:
            pix = doc[p_idx].get_pixmap(dpi=120)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # YOLO 推理
            res = self.layout_model.predict(img, imgsz=1024, conf=0.4, verbose=False)
            
            for box in res[0].boxes:
                cls = int(box.cls[0])
                if cls in [3, 5]: # Figure or Table
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    crop = img[coords[1]:coords[3], coords[0]:coords[2]]
                    if crop.size == 0: continue
                    
                    ocr_res, _ = self.ocr_engine(crop)
                    if ocr_res:
                        if cls == 5:
                            visual_metadata.append(f"#### 📊 Table (Page {p_idx+1})\n{self.table_to_md(ocr_res)}")
                        else:
                            labels = " ".join([x[1] for x in ocr_res])
                            visual_metadata.append(f"#### 🖼️ Figure (Page {p_idx+1}) Labels: {labels}")

        perf['2_visual_ocr'] = time.perf_counter() - t2

        # --- Step 4: 语义切片与无关项剔除 ---
        t3 = time.perf_counter()
        
        front_clean = page_texts[0] + "\n" + page_texts[1]
        
        method_clean = "[Methodology Not Detected]"
        if idx_method != -1:
            method_clean = page_texts[idx_method][:2500]

        conc_start_page = idx_conc if idx_conc != -1 else max(0, idx_ref - 1)
        conc_pool = "\n".join(page_texts[conc_start_page : idx_ref+1])
        
        # 【修复点】：直接使用编译好的正则对象的 split 方法，不传 flags
        temp_text = re_ref.split(conc_pool, maxsplit=1)[0]
        
        rw_match = re_related.search(temp_text)
        if rw_match:
            temp_text = temp_text[:rw_match.start()]
            
        conclusion_clean = temp_text[-2500:].strip()

        perf['3_slicing'] = time.perf_counter() - t3

        # --- Step 5: 导出 ---
        t4 = time.perf_counter()
        
        md_content = f"""# Analysis: {base_name}
## 1. Front Matter
{front_clean}
## 2. Methodology
{method_clean}
## 3. Conclusion (Cleaned)
{conclusion_clean}
---
## 4. Visual Evidence
{chr(10).join(visual_metadata)}
"""
        prompt_content = f"""<|im_start|>system
You are a research assistant.
<|im_end|>
<|im_start|>user
[FRONT]
{front_clean[:3000]}
[VISUAL]
{chr(10).join(visual_metadata).replace('#### ', '')}
[CONCLUSION]
{conclusion_clean}
Task: Summarize contributions and metrics.
<|im_end|>
<|im_start|>assistant"""

        with open(os.path.join(out_folder, f"{base_name}_report.md"), "w", encoding="utf-8") as f: f.write(md_content)
        with open(os.path.join(out_folder, f"{base_name}_prompt.txt"), "w", encoding="utf-8") as f: f.write(prompt_content)

        perf['4_io'] = time.perf_counter() - t4
        doc.close()
        return perf

if __name__ == "__main__":
    MODEL_PATH = "models/doclayout_yolo_docstructbench_imgsz1024.pt"
    engine = OptimizedAcademicEngine(MODEL_PATH)
    test_pdf = "osdi2025/osdi25-adam.pdf"
    if os.path.exists(test_pdf):
        perf = engine.process_pdf(test_pdf)
        print("\n--- Summary ---")
        for k, v in perf.items(): print(f"{k:20}: {v:.4f}s")