import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# 读取测试数据
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

# ======== 参数设置 ========
num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # ✅ GPU 优先
use_gpu = torch.cuda.is_available()

# ======== 模型定义 ========
class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=0
        )

        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs

# ======== 主程序 ========
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_glove.pickle3')
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word, vocab] = \
        pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    # 将所有张量移到 GPU
    train_features, train_labels = train_features.to(device), train_labels.to(device)
    val_features, val_labels = val_features.to(device), val_labels.to(device)
    test_features = test_features.to(device)
    weight = weight.to(device)

    net = SentimentNet(embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features,)

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ======== 训练循环 ========
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0

        net.train()
        with tqdm(total=len(train_iter), desc=f'Epoch {epoch}') as pbar:
            for feature, label in train_iter:
                n += 1
                feature, label = feature.to(device), label.to(device)
                net.zero_grad()

                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()

                train_acc += accuracy_score(torch.argmax(score.detach(), dim=1).cpu(), label.cpu())
                train_loss += loss.item()

                pbar.set_postfix({
                    'train_loss': f'{train_loss / n:.4f}',
                    'train_acc': f'{train_acc / n:.2f}'
                })
                pbar.update(1)

        # ======== 验证 ========
        net.eval()
        with torch.no_grad():
            for val_feature, val_label in val_iter:
                m += 1
                val_feature, val_label = val_feature.to(device), val_label.to(device)
                val_score = net(val_feature)
                loss = loss_function(val_score, val_label)
                val_acc += accuracy_score(torch.argmax(val_score, dim=1).cpu(), val_label.cpu())
                val_loss += loss.item()

        end = time.time()
        print(f"Epoch {epoch}: Train Loss={train_loss/n:.4f}, Train Acc={train_acc/n:.2f}, "
              f"Val Loss={val_loss/m:.4f}, Val Acc={val_acc/m:.2f}, Time={end-start:.2f}s")

    # ======== 预测 ========
    test_pred = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for (test_feature,) in test_iter:
                test_feature = test_feature.to(device)
                test_score = net(test_feature)
                test_pred.extend(torch.argmax(test_score, dim=1).cpu().numpy().tolist())
                pbar.update(1)

    # ======== 保存结果 ========
    os.makedirs('./result', exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/lstm_gpu.csv", index=False, quoting=3)
    logging.info('result saved!')
