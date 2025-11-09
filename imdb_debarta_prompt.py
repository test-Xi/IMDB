import os
import sys
import logging
import inspect
import pandas as pd
import numpy as np
import torch
import datasets

from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import PromptTuningConfig, get_peft_model, TaskType

# ---- 环境变量：抑制无关日志 & 禁止遥测 ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ---- 日志配置 ----
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
logging.root.setLevel(logging.INFO)
logger.info(f"running {program}")

# ---- 固定随机种子 ----
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---- 数据加载 ----
train = pd.read_csv(
    "./corpus/imdb/labeledTrainData.tsv",
    header=0,
    delimiter="\t",
    quoting=3
)
test = pd.read_csv(
    "./corpus/imdb/testData.tsv",
    header=0,
    delimiter="\t",
    quoting=3
)
logger.info(f"Loaded train={len(train)} rows, test={len(test)} rows.")

train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
logger.info(f"Split: train={len(train_df)}, val={len(val_df)}")

# 显式转成 list，避免类型歧义
train_ds = datasets.Dataset.from_dict({
    "label": train_df["sentiment"].tolist(),
    "text":  train_df["review"].tolist()
})
val_ds = datasets.Dataset.from_dict({
    "label": val_df["sentiment"].tolist(),
    "text":  val_df["review"].tolist()
})
test_ds = datasets.Dataset.from_dict({
    "text": test["review"].tolist()
})

# ---- 模型与分词器：本地加载 ----
model_id = r"./models/deberta-v3-base"   # 确认路径正确
logger.info(f"Loading model and tokenizer from {model_id}")

tokenizer = DebertaV2Tokenizer.from_pretrained(
    model_id,
    local_files_only=True,   # ✅ 只用本地
)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,       # 足够 IMDB，用 256 提升速度
    )

tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_val = val_ds.map(preprocess_function, batched=True)
tokenized_test = test_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---- 基础分类模型（本地） ----
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    local_files_only=True,
)

# ---- Prompt Tuning 配置 ----
# 这里是“软提示 / Prompt Tuning”，和前一个 P-Tuning 一致风格
peft_config = PromptTuningConfig(
    num_virtual_tokens=10,
    task_type=TaskType.SEQ_CLS,
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# ---- 本地 accuracy 计算（不再用 evaluate.load）----
def compute_metrics(eval_pred):
    # 兼容不同返回形式
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).astype(np.float32).mean().item()
    return {"accuracy": acc}

# ---- 自动过滤 TrainingArguments 支持参数 ----
def build_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return TrainingArguments(**valid)

training_args = build_training_args(
    output_dir="./checkpoint_prompt",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.05,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
)

# ---- Trainer ----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---- 训练 ----
trainer.train()

# ---- 验证并输出准确率 ----
logger.info("Evaluating validation set ...")
eval_result = trainer.evaluate(eval_dataset=tokenized_val)
acc = eval_result.get("eval_accuracy", None)
loss = eval_result.get("eval_loss", None)
if acc is not None and loss is not None:
    logger.info(f"Validation Accuracy: {acc:.4f}, Eval Loss: {loss:.4f}")
else:
    logger.info(f"Evaluation metrics: {eval_result}")

# ---- 预测与保存 ----
logger.info("Predicting test set ...")
preds = trainer.predict(tokenized_test)
logits = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
test_pred = np.argmax(logits, axis=-1).flatten()

os.makedirs("./result", exist_ok=True)
out_path = "./result/deberta_prompt.csv"
pd.DataFrame({"id": test["id"], "sentiment": test_pred}).to_csv(
    out_path,
    index=False,
    quoting=3,
)
logger.info(f"Result saved -> {out_path}")
