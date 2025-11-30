import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 AutoDL 上不要手动屏蔽 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import sys
import pandas as pd
import torch

from unsloth import FastLanguageModel
from peft import PeftModel  # 如果报 ModuleNotFoundError 就: pip install peft

# ---------------- 日志 ----------------
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(
    format='%(asctime)s: %(levelname)s: %(message)s',
    level=logging.INFO,
)
logger.info("Running %s", program)

# 推理用的 prompt（和训练时保持一致）
INFERENCE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of the following movie review. Respond with 'positive' or 'negative'.

### Input:
{}

### Response:
"""

# 你想用哪个 checkpoint 做推理，自己选一个
CHECKPOINT_DIR = "./outputs_qwen/checkpoint-3750"   # 建议用最后一个
BASE_MODEL_NAME = "unsloth/qwen3-4b-base-unsloth-bnb-4bit"


if __name__ == "__main__":
    # ------------- 1. 读取 test 数据 -------------
    logger.info("Loading test data...")
    test_path = "./corpus/imdb/testData.tsv"
    test_df = pd.read_csv(test_path, header=0, delimiter="\t", quoting=3)
    test_texts = test_df["review"].tolist()
    test_ids = test_df["id"].tolist()

    # ------------- 2. 加载底座模型 + tokenizer -------------
    logger.info("Loading base model: %s", BASE_MODEL_NAME)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 2048,
        dtype = None,
    )
    eos_token_id = tokenizer.eos_token_id

    # ------------- 3. 加载 LoRA 适配器 -------------
    logger.info("Loading LoRA adapter from %s", CHECKPOINT_DIR)
    # 把训练好的 LoRA 权重挂到底座模型上
    model = PeftModel.from_pretrained(
        model,
        CHECKPOINT_DIR,
    )

    # 切换到推理模式（unsloth 的优化）
    FastLanguageModel.for_inference(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device = %s", device)

    # ------------- 4. 逐条推理 -------------
    logger.info("Starting inference on test set...")
    predictions = []

    model.to(device)
    model.eval()

    for idx, review_text in enumerate(test_texts):
        prompt = INFERENCE_PROMPT.format(review_text)
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,             # 只需要输出 "positive"/"negative"
                eos_token_id=eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 从完整输出里截取 "### Response:" 后面的部分
        try:
            response_part = generated_text.split("### Response:")[1].strip().lower()
        except (IndexError, AttributeError):
            logger.warning("Failed to parse model output: %s", generated_text)
            response_part = ""

        if "positive" in response_part:
            predictions.append(1)
        elif "negative" in response_part:
            predictions.append(0)
        else:
            # 实在看不出来就保守当负样本
            logger.warning("Unknown sentiment, default to 0. Output: %s", response_part)
            predictions.append(0)

        if (idx + 1) % 500 == 0:
            logger.info("Processed %d / %d", idx + 1, len(test_texts))

    # ------------- 5. 保存 csv -------------
    logger.info("Saving predictions to CSV...")
    os.makedirs("./results", exist_ok=True)   # 注意这里用 ./results，不再用 ../results

    result_output = pd.DataFrame({
        "id": test_ids,
        "sentiment": predictions,
    })

    out_path = "./results/qwen3_4b_instruct_unsloth_infer_only.csv"
    result_output.to_csv(out_path, index=False, quoting=3)
    logger.info("Saved Kaggle submission file to: %s", out_path)
