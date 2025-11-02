import os
import sys
import logging
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # ✅ 新写法
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ✅ 使用 GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"🟢 Using device: {device}")

# ✅ 关闭不必要的 wandb 日志
os.environ["WANDB_DISABLED"] = "true"

# ✅ 读取数据
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)


# ✅ 自定义 Dataset
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 数据拆分
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train["review"].tolist(), train["sentiment"].tolist(), test_size=0.2, random_state=42
    )
    test_texts = test["review"].tolist()

    # ✅ 分词
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset = TestDataset(test_encodings)

    # ✅ 模型
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # ✅ DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ✅ 优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # ✅ 训练
    for epoch in range(3):
        model.train()
        start_time = time.time()
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs.logits, dim=1)
                acc = (preds == labels).float().mean().item()

                train_loss += loss.item()
                train_acc += acc
                pbar.set_postfix({
                    "loss": f"{train_loss / (pbar.n+1):.4f}",
                    "acc": f"{train_acc / (pbar.n+1):.4f}"
                })
                pbar.update(1)

        # ✅ 验证
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                acc = (preds == labels).float().mean().item()
                val_acc += acc

        print(f"\nEpoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} "
              f"| Train Acc: {train_acc/len(train_loader):.4f} "
              f"| Val Loss: {val_loss/len(val_loader):.4f} "
              f"| Val Acc: {val_acc/len(val_loader):.4f} "
              f"| Time: {time.time() - start_time:.2f}s")

    # ✅ 预测
    model.eval()
    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Predicting") as pbar:
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                test_pred.extend(preds.cpu().numpy().tolist())
                pbar.update(1)

    # ✅ 保存结果
    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/bert_native.csv", index=False, quoting=3)
    logging.info("✅ Result saved to ./result/bert_native.csv")
