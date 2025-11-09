import os
import sys
import logging
import inspect
import pandas as pd
import numpy as np
import datasets

# ---- 抑制 TF 噪声 & HF 遥测 ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# LoRA / PEFT
from peft import LoraConfig, get_peft_model, TaskType

# ---- 日志 ----
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
logging.root.setLevel(logging.INFO)
logger.info(f"running {program}")

# ---- 指标：优先 evaluate，失败回退 datasets.load_metric ----
try:
    import evaluate  # type: ignore
    _acc_metric = evaluate.load("accuracy")
    def _compute_acc(preds, refs):
        return {"accuracy": _acc_metric.compute(predictions=preds, references=refs)["accuracy"]}
except Exception:
    _acc_metric = datasets.load_metric("accuracy")
    def _compute_acc(preds, refs):
        return _acc_metric.compute(predictions=preds, references=refs)

# ---- 数据 ----
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test  = pd.read_csv("./corpus/imdb/testData.tsv",     header=0, delimiter="\t", quoting=3)
logger.info(f"Loaded train={len(train)} rows, test={len(test)} rows.")

train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
logger.info(f"Split: train={len(train_df)}, val={len(val_df)}")

train_ds = datasets.Dataset.from_dict({"label": train_df["sentiment"], "text": train_df["review"]})
val_ds   = datasets.Dataset.from_dict({"label": val_df["sentiment"],   "text": val_df["review"]})
test_ds  = datasets.Dataset.from_dict({"text": test["review"]})

# ---- 模型 & 分词器 ----
model_id = "microsoft/deberta-v3-base"
logger.info(f"Loading model and tokenizer from {model_id} ...")
tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)

def preprocess_function(examples):
    # 512 对 IMDB 足够；旧环境里 DataCollatorWithPadding 会自动 pad
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_val   = val_ds.map(preprocess_function, batched=True)
tokenized_test  = test_ds.map(preprocess_function, batched=True)
data_collator   = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# ---- LoRA 轻量微调 ----
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---- 指标函数：兼容旧版 eval_pred 形态 ----
def compute_metrics(eval_pred):
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return _compute_acc(preds, labels)

# ---- 训练参数：自动过滤当前版本支持的参数 ----
def build_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return TrainingArguments(**valid)

training_args = build_training_args(
    output_dir="./checkpoint",
    num_train_epochs=1,                 # 演示跑 1 epoch，交作业可调大
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
)

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

# ---- 评估并输出准确率 ----
try:
    logger.info("Evaluating validation set ...")
    eval_result = trainer.evaluate(eval_dataset=tokenized_val)
    acc = eval_result.get("eval_accuracy", None)
    loss = eval_result.get("eval_loss", None)
    if acc is not None and loss is not None:
        logger.info(f"Validation Accuracy: {acc:.4f}, Eval Loss: {loss:.4f}")
    else:
        logger.info(f"Evaluation metrics: {eval_result}")
except Exception as e:
    logger.warning(f"evaluate() skipped: {e}")

# ---- 预测 & 保存 ----
logger.info("Predicting test set ...")
preds = trainer.predict(tokenized_test)
test_pred = np.argmax(preds.predictions, axis=-1).flatten()

os.makedirs("./result", exist_ok=True)
out_path = "./result/deberta_lora.csv"
pd.DataFrame({"id": test["id"], "sentiment": test_pred}).to_csv(out_path, index=False, quoting=3)
logger.info(f"Result saved -> {out_path}")
