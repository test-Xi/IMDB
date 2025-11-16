import os
import sys
import logging
import datasets  # 只用来构建 Dataset
import losses

import torch.nn as nn

import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split
# 本地 bert 模型路径
LOCAL_BERT_DIR = os.path.join(
    os.path.dirname(__file__),
    "models",
    "bert-base-uncased"
)
# ================== 读数据 ==================
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)


class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = 0.2  # 对比损失的权重

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # 交叉熵损失：训练 + 验证都要用
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            scl_loss = 0.0

            # 只在训练阶段启用 SupConLoss，避免 eval 出现 NaN
            if self.training and self.alpha > 0:
                scl_fct = losses.SupConLoss()
                # [bsz, hidden] -> [bsz, 1, hidden]，符合 SupConLoss 的输入格式
                features = pooled_output.unsqueeze(1)

                # 检查 batch 里是否有“至少 2 个样本”的类别，避免除零
                unique_labels, counts = labels.unique(return_counts=True)
                if (counts > 1).any():
                    scl_loss = scl_fct(features, labels)

            loss = ce_loss + self.alpha * scl_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % " ".join(sys.argv))

    # ============= 划分训练/验证 =============
    train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

    train_dict = {"label": train_df["sentiment"], "text": train_df["review"]}
    val_dict = {"label": val_df["sentiment"], "text": val_df["review"]}
    test_dict = {"text": test["review"]}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    tokenizer = BertTokenizerFast.from_pretrained(LOCAL_BERT_DIR)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertScratch.from_pretrained(LOCAL_BERT_DIR, num_labels=2)

    # ============= 指标=============
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = (predictions == labels).mean().item()
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir="./checkpoint",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        eval_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ============= 预测 test 集 =============
    prediction_outputs = trainer.predict(tokenized_test)
    # 官方返回 PredictionOutput(predictions, label_ids, metrics)
    test_pred = np.argmax(prediction_outputs.predictions, axis=-1).flatten()
    print(test_pred)

    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/bert_scratch.csv", index=False, quoting=3)
    logging.info("result saved!")