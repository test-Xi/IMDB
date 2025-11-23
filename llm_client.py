# llm_client.py
# 使用 Qwen2.5-1.5B-Instruct 作为指令大模型，封装成一个 IMDB 情感分类“客户端”
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 可以根据需要换成别的指令模型
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def build_imdb_prompt(review_text: str) -> str:
    """
    给一条 IMDB 影评，构造一个中文说明的 Instruction 风格 prompt。
    模型任务：只输出 0 或 1。
    """
    return f"""下面是一项任务的说明以及对应的输入文本，请根据要求完成任务。

### 指令（Instruction）:
你是一名电影评论情感分析专家。现在给你一条英文电影评论，请判断这条评论的情感是“积极”还是“消极”。
请只输出一个数字：积极输出 1，消极输出 0。不要输出任何其它内容，也不要解释。

### 输入（Input）:
{review_text}

### 回答（Response）:
"""


def parse_label_from_output(output: str) -> int:
    """
    从模型输出文本中解析 0/1。
    简单做法：取第一行，看看是 "0" 还是 "1"。
    """
    text = output.strip()
    first_line = text.splitlines()[0].strip()

    if first_line == "0":
        return 0
    if first_line == "1":
        return 1

    # 容错：如果第一行里包含 0 或 1
    if "0" in first_line and "1" not in first_line:
        return 0
    if "1" in first_line and "0" not in first_line:
        return 1

    raise ValueError(f"无法从输出中解析标签: {output!r}")


class LLMClient:
    """
    指令大模型客户端：
    - 负责加载 tokenizer + model
    - 提供 generate() 和 predict_imdb_label() 接口
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = 8,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        print(f"[LLMClient] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
        )

        if torch.cuda.is_available():
            torch_dtype = torch.float16
            print("[LLMClient] Using CUDA with float16")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # 没有 GPU 时退回 CPU（会很慢）
            torch_dtype = torch.float32
            print("[LLMClient] CUDA not available, using CPU with float32 (may be very slow)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        # 有些模型没有 pad_token，保险起见设置一下
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        """
        给定 prompt，生成文本。
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # 不随机，更稳定
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 只取新生成的部分（去掉提示词）
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def predict_imdb_label(self, review_text: str) -> int:
        """
        对一条 IMDB 影评进行情感分类，返回 0/1。
        """
        prompt = build_imdb_prompt(review_text)
        output = self.generate(prompt)
        label = parse_label_from_output(output)
        return int(label)


if __name__ == "__main__":
    # 简单自测
    client = LLMClient()
    review = "This movie was incredibly entertaining and emotionally touching. I really enjoyed it."
    label = client.predict_imdb_label(review)
    print("Pred label:", label)  # 预期为 1（积极）
