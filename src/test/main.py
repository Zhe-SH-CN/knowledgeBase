import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import fitz  # PyMuPDF
import time
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ================= 配置区 =================
# API Keys从文件读取
API_KEYS_FILE = "./siliconflow-apikeys.txt"

# HuggingFace模型配置 - 使用Qwen3-VL-4B支持多模态任务（文本+图片）
HF_MODEL_NAME = (
    "Qwen/Qwen3-4B-Instruct-2507"  # 4B参数量，支持视觉+文本，适合Jetson AGX 64GB
)

INPUT_DIR = "./osdi2025"
OUTPUT_BASE_DIR = "./paper_states"
METADATA_DIR = os.path.join(OUTPUT_BASE_DIR, "metadata")
TAG_DIR = os.path.join(OUTPUT_BASE_DIR, "tag")

# 创建输出目录
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(TAG_DIR, exist_ok=True)


# 读取API Keys
def load_api_keys():
    """从文件读取API Keys"""
    with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    return keys


API_KEYS = load_api_keys()


# ================= 本地模型管理器 =================
class LocalModelManager:
    def __init__(self, model_name):
        """初始化本地模型"""
        print(f"正在加载模型: {model_name}")

        # 检查CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # 使用第一个GPU
            print(f"使用设备: CUDA:0 (GPU)")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )
        else:
            self.device = torch.device("cpu")
            print(f"警告: CUDA不可用，使用CPU运行（速度会较慢）")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        self.model = self.model.to(self.device)

        self.model.eval()
        print("模型加载完成\n")

    def generate_response(self, prompt, system_prompt="", max_new_tokens=2000):
        """生成回复"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        input_token_count = model_inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        output_token_count = len(generated_ids[0])
        total_tokens = input_token_count + output_token_count

        return response, total_tokens


# ================= 论文处理类 =================
class PaperMetadataGenerator:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.processed_count = 0
        self.total_tokens_metadata = 0
        self.total_tokens_tag = 0

    def extract_text_from_pdf(self, pdf_path, max_pages=8):
        """从PDF中提取前几页的文本"""
        doc = fitz.open(pdf_path)
        text_parts = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)

    def generate_metadata(self, text_content):
        """使用LLM生成metadata"""
        prompt = f"""请分析以下论文文本,提取元数据信息并以JSON格式返回。

论文文本:
{text_content[:8000]}

请返回以下格式的JSON(只返回JSON,不要其他内容):
{{
  "title": "论文标题",
  "authors": ["作者1", "作者2"],
  "abstract": "论文摘要(如果有)",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "category": "主要类别(如:系统、网络、安全、机器学习等)",
  "key_contributions": ["主要贡献1", "主要贡献2"]
}}"""

        system_prompt = "你是一个专业的学术论文分析助手,擅长提取论文的元数据信息。"
        result_text, token_usage = self.model_manager.generate_response(
            prompt, system_prompt
        )

        # 尝试提取JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        try:
            metadata = json.loads(result_text)
        except json.JSONDecodeError:
            metadata = {
                "title": "未能提取",
                "authors": [],
                "abstract": "",
                "keywords": [],
                "category": "未分类",
                "key_contributions": [],
            }

        return metadata, token_usage

    def generate_tags(self, text_content):
        """使用LLM生成tags"""
        prompt = f"""请分析以下论文文本,提取标签和研究领域信息并以JSON格式返回。

论文文本:
{text_content[:8000]}

请返回以下格式的JSON(只返回JSON,不要其他内容):
{{
  "tags": ["标签1", "标签2", "标签3", "标签4"],
  "research_areas": ["研究领域1", "研究领域2", "研究领域3"]
}}"""

        system_prompt = "你是一个专业的学术论文分析助手,擅长提取论文的标签和研究领域。"
        result_text, token_usage = self.model_manager.generate_response(
            prompt, system_prompt
        )

        # 尝试提取JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        try:
            tags_info = json.loads(result_text)
        except json.JSONDecodeError:
            tags_info = {"tags": [], "research_areas": []}

        return tags_info, token_usage

    def process_paper(self, pdf_path, pbar):
        """处理单篇论文"""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        metadata_save_path = os.path.join(METADATA_DIR, f"{base_name}_metadata.json")
        tag_save_path = os.path.join(TAG_DIR, f"{base_name}_tag.json")

        # 检查是否已存在
        metadata_exists = os.path.exists(metadata_save_path)
        tag_exists = os.path.exists(tag_save_path)

        if metadata_exists and tag_exists:
            pbar.set_description(f"[跳过] {base_name}")
            return

        pbar.set_description(f"[处理] {base_name}")

        try:
            # 1. 提取PDF文本
            text_content = self.extract_text_from_pdf(pdf_path)

            if len(text_content.strip()) < 100:
                pbar.write(f"  [警告] {base_name} 文本内容过少")

            # 2. 生成metadata
            if not metadata_exists:
                metadata, token_usage_metadata = self.generate_metadata(text_content)
                self.total_tokens_metadata += token_usage_metadata

                metadata_result = {
                    "filename": base_name,
                    "file_path": pdf_path,
                    "metadata": metadata,
                    "token_usage": token_usage_metadata,
                }

                with open(metadata_save_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_result, f, ensure_ascii=False, indent=2)

            # 3. 生成tags
            if not tag_exists:
                tags_info, token_usage_tag = self.generate_tags(text_content)
                self.total_tokens_tag += token_usage_tag

                tag_result = {
                    "filename": base_name,
                    "file_path": pdf_path,
                    "tags_info": tags_info,
                    "token_usage": token_usage_tag,
                }

                with open(tag_save_path, "w", encoding="utf-8") as f:
                    json.dump(tag_result, f, ensure_ascii=False, indent=2)

            self.processed_count += 1

        except Exception as e:
            pbar.write(f"  [错误] {base_name}: {e}")

    def get_directory_size(self, directory):
        """计算目录的总大小（字节）"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size

    def format_size(self, size_bytes):
        """格式化文件大小"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def save_stats(self, directory, elapsed_time, total_tokens):
        """保存统计信息"""
        dir_size = self.get_directory_size(directory)
        stats = {
            "total_files": len(
                [f for f in os.listdir(directory) if f.endswith(".json")]
            ),
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": f"{elapsed_time:.2f}秒 ({elapsed_time / 60:.2f}分钟)",
            "total_tokens": total_tokens,
            "disk_size_bytes": dir_size,
            "disk_size_formatted": self.format_size(dir_size),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        stats_path = os.path.join(directory, "_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return stats

    def batch_process(self):
        """批量处理所有论文"""
        pdf_files = [
            os.path.join(INPUT_DIR, f)
            for f in sorted(os.listdir(INPUT_DIR))
            if f.lower().endswith(".pdf")
        ]

        total = len(pdf_files)
        print(f"找到 {total} 篇论文，开始处理...\n")

        # 记录整体开始时间
        overall_start_time = time.time()

        # 记录metadata处理开始时间
        metadata_start_time = time.time()

        with tqdm(total=total, desc="处理论文", unit="篇") as pbar:
            for pdf_path in pdf_files:
                self.process_paper(pdf_path, pbar)
                pbar.update(1)

        # 计算总体耗时
        overall_elapsed_time = time.time() - overall_start_time

        print(f"\n处理完成！共处理 {self.processed_count} 篇论文")
        print(
            f"总耗时: {overall_elapsed_time:.2f}秒 ({overall_elapsed_time / 60:.2f}分钟)"
        )

        # 保存metadata统计
        print("\n正在计算metadata统计信息...")
        metadata_stats = self.save_stats(
            METADATA_DIR, overall_elapsed_time, self.total_tokens_metadata
        )
        print(
            f"Metadata: {metadata_stats['total_files']}个文件, "
            f"{metadata_stats['disk_size_formatted']}, "
            f"Token使用: {metadata_stats['total_tokens']}"
        )

        # 保存tag统计
        print("\n正在计算tag统计信息...")
        tag_stats = self.save_stats(
            TAG_DIR, overall_elapsed_time, self.total_tokens_tag
        )
        print(
            f"Tag: {tag_stats['total_files']}个文件, "
            f"{tag_stats['disk_size_formatted']}, "
            f"Token使用: {tag_stats['total_tokens']}"
        )

        print(f"\n总Token使用量: {self.total_tokens_metadata + self.total_tokens_tag}")
        print(f"结果保存在: {OUTPUT_BASE_DIR}")


# ================= 主程序 =================
if __name__ == "__main__":
    print("=" * 60)
    print("论文元数据和标签提取系统")
    print("=" * 60)

    # 初始化模型
    model_manager = LocalModelManager(HF_MODEL_NAME)

    # 初始化处理器
    generator = PaperMetadataGenerator(model_manager)

    # 批量处理
    generator.batch_process()

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
