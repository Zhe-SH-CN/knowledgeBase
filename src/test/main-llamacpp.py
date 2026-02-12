import os
import json
import time
import pymupdf4llm 
from tqdm import tqdm
import pickle
from llama_cpp import Llama
import re
from bs4 import BeautifulSoup

# ================= 环境变量与配置 =================
# 必须在 import 之后，但在模型实例化之前（虽然最好是编译时处理）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

INPUT_DIR = "./osdi2025"
OUTPUT_BASE_DIR = "./paper_states_nano"
METADATA_DIR = os.path.join(OUTPUT_BASE_DIR, "metadata")
TAG_DIR = os.path.join(OUTPUT_BASE_DIR, "tag")
CACHE_DIR = os.path.join(OUTPUT_BASE_DIR, "kv_cache")

# 模型路径
GGUF_MODEL_PATH = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"

for d in [METADATA_DIR, TAG_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
class LlamaNanoManager:
    def __init__(self, model_path):
        print(f"正在加载模型: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=32768,
            verbose=False 
        )
        print("模型加载完成。")

    def analyze_paper(self, compressed_text, state_path):
        # 优化 Prompt：明确要求处理转义字符，并给出一个明确的 JSON 结构
        full_prompt = f"<|im_start|>system\n你是一个学术分析专家。请严格按 JSON 格式输出结果，确保字符串内没有未转义的双引号。不要输出任何解释文字。<|im_end|>\n<|im_start|>user\n以下是论文的关键部分：\n{compressed_text}\n\n请提取元数据（metadata）和标签（tags_info）。\n\n注意：摘要部分请精简到100字以内。JSON 格式如下：\n{{\"metadata\": {{\"title\": \"...\", \"authors\": [], \"abstract\": \"...\", \"category\": \"...\"}}, \"tags_info\": {{\"tags\": [], \"research_areas\": []}}}}\n<|im_end|>\n<|im_start|>assistant\n"

        start_time = time.time()
        
        # 1. 加载 KV Cache
        # if os.path.exists(state_path):
        #     try:
        #         with open(state_path, "rb") as f:
        #             state_object = pickle.load(f)
        #             self.llm.load_state(state_object)
        #     except Exception: pass
        
        # 2. 推理 - 适当增加 max_tokens 防止截断
        output = self.llm(
            full_prompt,
            max_tokens=2048, 
            temperature=0.1,
            stop=["<|im_end|>", "观察"]
        )
        
        # 3. 保存 KV Cache
        # if not os.path.exists(state_path):
        #     try:
        #         state_object = self.llm.save_state()
        #         with open(state_path, "wb") as f:
        #             pickle.dump(state_object, f)
        #     except Exception: pass

        duration = time.time() - start_time
        res_text = output["choices"][0]["text"]
        
        return res_text, output["usage"], duration

class PaperProcessor:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def clean_json_text(self, text):
        """
        极其鲁棒的 JSON 提取逻辑
        """
        # 1. 去掉 Markdown 的代码块标记
        text = re.sub(r'```json\s*|\s*```', '', text)
        
        # 2. 尝试匹配第一个 { 和最后一个 } 之间的内容
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            text = match.group(1)
        
        # 3. 处理常见的 JSON 换行/转义问题
        text = text.replace('\n', ' ').replace('\r', '')
        return text

    def process_all(self):
        pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
        for filename in tqdm(pdf_files):
            base_name = os.path.splitext(filename)[0]
            m_path = os.path.join(METADATA_DIR, f"{base_name}_metadata.json")
            t_path = os.path.join(TAG_DIR, f"{base_name}_tag.json")
            s_path = os.path.join(CACHE_DIR, f"{base_name}.state")

            if os.path.exists(m_path) and os.path.exists(t_path): continue

            try:
                # 提取文本 (2000 + 1000 逻辑)
                md_text = pymupdf4llm.to_markdown(os.path.join(INPUT_DIR, filename))
                sliced_text = md_text if len(md_text) <= 3000 else f"{md_text[:2000]}\n...\n{md_text[-1000:]}"

                raw_res, usage, dur = self.model_manager.analyze_paper(sliced_text, s_path)
                
                # 清洗 JSON
                clean_res = self.clean_json_text(raw_res)
                
                try:
                    data = json.loads(clean_res)
                except json.JSONDecodeError as e:
                    # 如果解析失败，记录错误并跳过，方便后续排查
                    print(f"\n[Error] JSON解析失败: {filename}")
                    print(f"Raw Output: {raw_res[:200]}...") # 打印前200字查错
                    continue

                # 分开保存
                with open(m_path, "w", encoding="utf-8") as f:
                    json.dump({"metadata": data.get("metadata"), "proc_time": dur}, f, ensure_ascii=False, indent=2)
                with open(t_path, "w", encoding="utf-8") as f:
                    json.dump({"tags_info": data.get("tags_info"), "proc_time": dur}, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"处理失败 {filename}: {e}")

if __name__ == "__main__":
    # 确保模型文件存在
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"错误：找不到模型文件 {GGUF_MODEL_PATH}")
    else:
        manager = LlamaNanoManager(GGUF_MODEL_PATH)
        processor = PaperProcessor(manager)
        processor.process_all()