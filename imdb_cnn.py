import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# ========== 参数部分 ==========
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_filter = 128
filter_size = 3
batch_size = 64
labels = 2
lr = 0.8

# ✅ 改为 CPU 设备
device = torch.device('cpu')
use_gpu = False


# ========== 模型定义 ==========
class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_filter, filter_size, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.use_gpu = use_gpu
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

        self.conv1d = nn.Conv1d(embed_size, num_filter, filter_size, padding=1)
        self.activate = F.relu
        self.decoder = nn.Linear(num_filter, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        convolution = self.activate(self.conv1d(embeddings.permute([0, 2, 1])))
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        outputs = self.decoder(pooling.squeeze(dim=2))
        return outputs


# ========== 主程序 ==========
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ' '.join(sys.argv))

    # 加载数据
    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_glove.pickle3')
    [train_features, train_labels, val_features, val_labels, test_features,
     weight, word_to_idx, idx_to_word, vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    # 模型初始化
    net = SentimentNet(embed_size=embed_size, num_filter=num_filter, filter_size=filter_size,
                       weight=weight, labels=labels, use_gpu=use_gpu)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # 数据加载器
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ========== 训练阶段 ==========
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0

        with tqdm(total=len(train_iter), desc=f"Epoch {epoch}") as pbar:
            for feature, label in train_iter:
                n += 1
                feature = Variable(feature.to(device))
                label = Variable(label.to(device))

                net.zero_grad()
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()

                train_acc += accuracy_score(torch.argmax(score.detach(), dim=1), label)
                train_loss += loss.item()

                pbar.set_postfix({
                    'train_loss': f"{train_loss / n:.4f}",
                    'train_acc': f"{train_acc / n:.2f}"
                })
                pbar.update(1)

        # 验证阶段
        with torch.no_grad():
            for val_feature, val_label in val_iter:
                m += 1
                val_feature = val_feature.to(device)
                val_label = val_label.to(device)
                val_score = net(val_feature)
                val_loss = loss_function(val_score, val_label)
                val_acc += accuracy_score(torch.argmax(val_score.detach(), dim=1), val_label)
                val_losses += val_loss.item()

        end = time.time()
        print(f"\nEpoch {epoch}: Train Loss={train_loss / n:.4f}, "
              f"Train Acc={train_acc / n:.2f}, "
              f"Val Loss={val_losses / m:.4f}, "
              f"Val Acc={val_acc / m:.2f}, "
              f"Time={end - start:.2f}s")

    # ========== 测试与保存 ==========
    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for (test_feature,) in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                test_pred.extend(torch.argmax(test_score.detach(), dim=1).numpy().tolist())
                pbar.update(1)

    # 保存结果
    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/cnn.csv", index=False, quoting=3)
    logging.info('result saved!')
