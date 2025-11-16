import os

# 可选：如果服务器能访问 hf-mirror，可以保留；不能访问也无所谓
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 不强制绑 GPU 1，交给外面 CUDA_VISIBLE_DEVICES 配
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import unsloth
import torch
import sys
import logging

import pandas as pd
import numpy as np

from unsloth import FastModel, FastLanguageModel
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# ========== 数据读取 ==========
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

# 本地 DeBERTa 模型路径（按你上传到服务器的实际目录修改）
# 比如你放在 imdb_sentiment_analysis_torch/models/deberta-v3-base
LOCAL_DEBERTA_DIR = os.path.join(
    os.path.dirname(__file__),
    "models",
    "deberta-v3-base",   
)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ' '.join(sys.argv))

    # ========== 划分训练/验证（这里你只取了 20 条，小样本测试用） ==========
    train_df, val_df = train_test_split(train, test_size=.2, random_state=42)

    train_df, val_df = train_test_split(train, test_size=.2, random_state=42)
    test_df = test  # 全部测试

    train_dict = {'label': train_df["sentiment"], 'text': train_df['review']}
    val_dict = {'label': val_df["sentiment"], 'text': val_df['review']}
    test_dict = {"text": test_df['review']}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # ======== 使用本地 DeBERTa 模型 ========
    model_name = LOCAL_DEBERTA_DIR      # 本地路径
    NUM_CLASSES = 2

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        auto_model=AutoModelForSequenceClassification,
        num_labels=NUM_CLASSES,
        gpu_memory_utilization=0.8,  # 显存利用，OOM 再往下调
    )

    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        task_type="SEQ_CLS",
    )

    print("model parameters:" + str(sum(p.numel() for p in model.parameters())))


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = float((predictions == labels).mean())
        return {"accuracy": acc}

    # ========== Tokenize ==========
    def tokenize_function(examples):
        return tokenizer(examples['text'], max_length=512, truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ========== 训练参数 ==========
    training_args = TrainingArguments(
        output_dir="./checkpoint_deberta",     # 建议单独一个目录
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,

        warmup_steps=10,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",                  # 用字符串，避免 training_args.OptimizerNames 的兼容问题
        learning_rate=2e-5,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        num_train_epochs=3,
        save_strategy="epoch",

        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,

        label_names=["label"],
        report_to="none",                     # 关掉 wandb / tensorboard
    )

    from typing import Any, Dict, Tuple, Optional

    class UnslothSafeTrainer(Trainer):
        """避免调用 unsloth 注入的 prediction_step，直接用 model(**inputs)。"""

        def prediction_step(
            self,
            model,
            inputs: Dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: Optional[Tuple[str]] = None,
        ):
            inputs = self._prepare_inputs(inputs)

            labels = None
            if isinstance(inputs, dict):
                labels = inputs.get("labels", inputs.get("label", None))

            with torch.no_grad():
                outputs = model(**inputs)

            loss = None
            logits = None

            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
                logits = outputs.get("logits", outputs.get("predictions", None))
            elif isinstance(outputs, tuple):
                if len(outputs) == 1:
                    logits = outputs[0]
                elif len(outputs) >= 2:
                    a, b = outputs[0], outputs[1]
                    if getattr(a, "ndim", None) == 0 or (isinstance(a, torch.Tensor) and a.numel() == 1):
                        loss = a
                        logits = b
                    else:
                        logits = a

            if prediction_loss_only:
                return (loss, None, None)

            return (loss, logits, labels)

    trainer = UnslothSafeTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,   # 避免 tokenizer 警告
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("[DeBERTa Eval]", eval_metrics)
    # ========== 预测 test 集 ==========
    prediction_outputs = trainer.predict(test_dataset)
    print(prediction_outputs)

    test_pred = np.argmax(prediction_outputs.predictions, axis=-1).flatten()
    print(test_pred)

    # ========= 手动算一次验证集准确率 =========
    val_pred = trainer.predict(val_dataset)
    val_logits = val_pred.predictions
    val_labels = val_pred.label_ids

    val_pred_ids = np.argmax(val_logits, axis=-1)
    val_acc = float((val_pred_ids == val_labels).mean())
    print("[DeBERTa 手动验证集准确率]", val_acc)


    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test_df["id"], "sentiment": test_pred})
    result_output.to_csv("./result/deberta_unsloth.csv", index=False, quoting=3)
    logging.info('result saved!')
