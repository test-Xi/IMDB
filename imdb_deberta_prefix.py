import os
import sys
import logging
import inspect
import random
import numpy as np
import pandas as pd
import datasets
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import PrefixTuningConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

# ========== 环境与日志配置 ==========
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
logging.root.setLevel(logging.INFO)
logger.info(f"running {program}")

# ========== 固定随机种子 ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ========== 加载 IMDb 数据 ==========
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", delimiter="\t", quoting=3)
logger.info(f"Loaded train={len(train)}, test={len(test)}")

train = train.rename(columns={"sentiment": "label", "review": "text"})
test = test.rename(columns={"review": "text"})

train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
logger.info(f"Split: train={len(train_df)}, val={len(val_df)}")

train_ds = datasets.Dataset.from_pandas(train_df[["label", "text"]], preserve_index=False)
val_ds   = datasets.Dataset.from_pandas(val_df[["label", "text"]], preserve_index=False)
test_ds  = datasets.Dataset.from_pandas(test[["text"]], preserve_index=False)

# ========== 模型与分词器（本地 T5） ==========
model_id = "./models/t5-base"   # 本地路径
logger.info(f"Loading model and tokenizer from {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

# ========== 文本预处理 ==========
max_source_length = 256
max_target_length = 4

def preprocess_function(examples):
    inputs = [f"Review: {t}" for t in examples["text"]]
    model_inputs = tokenizer(
        inputs, truncation=True, max_length=max_source_length,
    )

    if "label" in examples:
        labels_str = [str(int(l)) for l in examples["label"]]
        labels = tokenizer(
            text_target=labels_str,
            truncation=True,
            max_length=max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
tokenized_val   = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)
tokenized_test  = test_ds.map(preprocess_function, batched=True, remove_columns=test_ds.column_names)

# ========== 模型 + Prefix Tuning ==========
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=True)

peft_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=30,
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ========== TrainingArguments ==========
def build_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return TrainingArguments(**valid)

training_args = build_training_args(
    output_dir="./checkpoint_prefix",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="no",
    evaluation_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=max_target_length,
    generation_num_beams=1,
    fp16=torch.cuda.is_available(),
)

# ========== 解码 & 本地 accuracy ==========
def decode_to_labels(ids_array):
    """
    彻底兼容 Hugging Face T5 输出结构。
    自动处理 [[[...]]] 或更深层嵌套，展开为二维数组后解码。
    """
    import itertools

    # 转为 Python list
    if isinstance(ids_array, torch.Tensor):
        ids_array = ids_array.tolist()

    # 展开所有嵌套直到最外层是样本，内层是 token id
    while isinstance(ids_array, list) and len(ids_array) > 0 and isinstance(ids_array[0], list):
        # 检查最内层是否为 int
        if len(ids_array[0]) > 0 and isinstance(ids_array[0][0], int):
            break
        ids_array = list(itertools.chain.from_iterable(ids_array))

    # 保险起见，再压成 numpy 数组再转回 list
    ids_array = np.array(ids_array).tolist()

    # 解码为文本
    texts = tokenizer.batch_decode(ids_array, skip_special_tokens=True)
    return [1 if t.strip().startswith("1") else 0 for t in texts]


def compute_metrics_fn(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    # labels 中的 -100 改成 pad 再解码
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    pred_labels = decode_to_labels(predictions)
    true_labels = decode_to_labels(labels)

    pred_arr = np.array(pred_labels, dtype=np.int32)
    true_arr = np.array(true_labels, dtype=np.int32)
    acc = (pred_arr == true_arr).astype(np.float32).mean().item()
    return {"accuracy": acc}

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,
)

# ========== 训练 ==========
trainer.train()

# ========== 验证 ==========
logger.info("Evaluating validation set ...")
eval_result = trainer.evaluate(eval_dataset=tokenized_val)
acc = eval_result.get("eval_accuracy", None)
loss = eval_result.get("eval_loss", None)
if acc is not None and loss is not None:
    logger.info(f"Validation Accuracy: {acc:.4f}, Eval Loss: {loss:.4f}")
else:
    logger.info(f"Evaluation metrics: {eval_result}")

# ========== 预测并保存 ==========
logger.info("Predicting test set ...")
preds = trainer.predict(tokenized_test)
pred_ids = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
test_pred = decode_to_labels(pred_ids)

os.makedirs("./result", exist_ok=True)
out_path = "./result/t5_prefix.csv"
pd.DataFrame({"id": test["id"], "sentiment": test_pred}).to_csv(out_path, index=False, quoting=3)
logger.info(f"Result saved -> {out_path}")
