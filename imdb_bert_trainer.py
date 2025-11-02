import os
import sys
import logging
import torch
import datasets
import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, BertForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# ✅ 强制使用 GPU（如可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🟢 Using device: {device.upper()}")

# 禁用 wandb
os.environ["WANDB_DISABLED"] = "true"

# 读取数据
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 1️⃣ 数据划分
    train, val = train_test_split(train, test_size=.2, random_state=42)

    train_dict = {'label': train["sentiment"].astype(int), 'text': train['review']}
    val_dict   = {'label': val["sentiment"].astype(int),   'text': val['review']}
    test_dict  = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset   = datasets.Dataset.from_dict(val_dict)
    test_dataset  = datasets.Dataset.from_dict(test_dict)

    # 2️⃣ 分词
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val   = val_dataset.map(preprocess_function, batched=True)
    tokenized_test  = test_dataset.map(preprocess_function, batched=True)

    if "label" in tokenized_train.column_names:
        tokenized_train = tokenized_train.rename_column("label", "labels")
    if "label" in tokenized_val.column_names:
        tokenized_val   = tokenized_val.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3️⃣ 模型（GPU 加载）
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # 4️⃣ 评估指标
    def compute_metrics(eval_pred):
        try:
            logits, labels = eval_pred
        except Exception:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        if labels is None:
            return {}
        acc = (preds == labels).astype(np.float32).mean()
        return {"accuracy": float(acc)}

    # 5️⃣ 训练参数
    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./result', exist_ok=True)

    training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        report_to=[],                # 禁用wandb
        fp16=True if device == "cuda" else False,  # ✅ 启用混合精度训练（GPU上更快）
    )

    # 6️⃣ Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7️⃣ 开始训练
    trainer.train()

    # 8️⃣ 测试集预测
    prediction_outputs = trainer.predict(tokenized_test)
    logits = prediction_outputs[0] if isinstance(prediction_outputs, tuple) else prediction_outputs.predictions
    test_pred = np.argmax(logits, axis=-1).astype(int).flatten()

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/bert_trainer.csv", index=False, quoting=3)
    logging.info('✅ Result saved to ./result/bert_trainer.csv')
