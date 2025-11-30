import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在 Colab 里不要手动屏蔽 GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import logging
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# --- 日志设置 ---
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
# logger.info(...) 移动到 main 内部

# --- Alpaca 指令模版 ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of the following movie review. Respond with 'positive' or 'negative'.

### Input:
{}

### Response:
{}"""


# --- 数据格式化函数 ---
def formatting_prompts_func(examples):
    inputs = examples["text"]
    labels = examples["label"]
    outputs_text = []

    global EOS_TOKEN
    if EOS_TOKEN is None:
        raise ValueError("EOS_TOKEN is not set. Make sure tokenizer is loaded first.")

    for input_text, label in zip(inputs, labels):
        label_text = "positive" if label == 1 else "negative"
        text = alpaca_prompt.format(input_text, label_text) + EOS_TOKEN
        outputs_text.append(text)

    return {"text": outputs_text}


# --- 主执行程序 ---
if __name__ == '__main__':

    logger.info(r"running %s" % ''.join(sys.argv))

    # --- 1. 加载数据 ---
    logger.info("Loading data...")
    train_df = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test_df = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

    train_df, val_df = train_test_split(train_df, test_size=.2, random_state=3407)

    train_dict = {'label': train_df["sentiment"], 'text': train_df['review']}
    val_dict = {'label': val_df["sentiment"], 'text': val_df['review']}
    test_dict = {"text": test_df['review']}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # --- 2. 加载模型和 Tokenizer ---
    logger.info("Loading Qwen model...")

    model_name = "unsloth/Qwen3-4B-base"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        max_seq_length=2048,
        dtype=None,
    )

    EOS_TOKEN = tokenizer.eos_token
    logger.info(f"EOS token set to: {EOS_TOKEN}")

    # --- 3. PEFT (LoRA) 设置 ---
    logger.info("Setting up PEFT...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
    )

    logger.info("Model parameters:" + str(sum(p.numel() for p in model.parameters())))

    # --- 4. 格式化数据集 ---
    logger.info("Formatting datasets...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True,)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True,)

    # --- 5. 训练参数 ---
    logger.info("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir="outputs_qwen",
        per_device_train_batch_size=16,
        # gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="epoch",
        # 这里不再传 evaluation_strategy，避免版本不兼容
        # evaluation_strategy="epoch",
    )

    # --- 6. 初始化 SFTTrainer ---
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    # --- 7. 训练 ---
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    print(trainer_stats)

    # --- 8. 推理 (Generation) ---
    logger.info("Starting inference...")
    FastLanguageModel.for_inference(model)

    inference_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of the following movie review. Respond with 'positive' or 'negative'.

### Input:
{}

### Response:
"""

    test_texts = test_dataset['text']
    test_ids = test_df['id']
    predictions = []

    for review_text in test_texts:
        prompt = inference_prompt.format(review_text)
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=2048).to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=3, eos_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            response_part = generated_text.split("### Response:")[1].strip().lower()

            if "positive" in response_part:
                predictions.append(1)
            elif "negative" in response_part:
                predictions.append(0)
            else:
                predictions.append(0)

        except (IndexError, AttributeError):
            logger.warning(f"Failed to parse model output: {generated_text}")
            predictions.append(0)

    # --- 9. 保存结果 ---
    logger.info("Saving results...")
    result_output = pd.DataFrame(data={"id": test_ids, "sentiment": predictions})
    result_output.to_csv("../results/qwen3_4b_instruct_unsloth.csv", index=False, quoting=3)
    logger.info('Qwen result saved!')
