import logging
import os
import sys
import time
import math
import re
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ================= 基本参数 ================= #
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 16   # 防止显存溢出
labels = 2
lr = 0.05

# ✅ 自动检测 GPU / CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
print(f"Using device: {device}")

# ================= 数据读取 ================= #
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    return ' '.join(words)

# ================= 词汇表 ================= #
class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()
        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, train, test, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in train:
            for token in sentence:
                token_freqs[token] += 1
        for sentence in test:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items()
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]

# ================= mask生成 ================= #
def length_to_mask(lengths):
    max_length = torch.max(lengths)
    mask = torch.arange(max_length, device=lengths.device).expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)
    return mask

# ================= 位置编码 ================= #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(0) > self.pe.size(0):
            repeat_times = math.ceil(x.size(0) / self.pe.size(0))
            pe = self.pe.repeat(repeat_times, 1, 1)[:x.size(0), :]
        else:
            pe = self.pe[:x.size(0), :]
        x = x + pe
        return x

# ================= Transformer模型 ================= #
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1,
                 max_len=6000, activation: str = "relu"):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(embedding_dim, num_class)

    def forward(self, inputs, lengths):
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

# ================= Dataset封装 ================= #
class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# ✅ 修复后的 collate_fn，兼容 train/val/test
def collate_fn(examples):
    MAX_LEN = 256
    if isinstance(examples[0], tuple):  # 训练或验证集 (tokens, label)
        lengths = torch.tensor([min(len(ex[0]), MAX_LEN) for ex in examples])
        inputs = [torch.tensor(ex[0][:MAX_LEN]) for ex in examples]
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        inputs = pad_sequence(inputs, batch_first=True)
        return inputs, lengths, targets
    else:  # 测试集 (只有 tokens)
        lengths = torch.tensor([min(len(ex), MAX_LEN) for ex in examples])
        inputs = [torch.tensor(ex[:MAX_LEN]) for ex in examples]
        inputs = pad_sequence(inputs, batch_first=True)
        return inputs, lengths

# ================= 主程序 ================= #
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    clean_train_reviews, train_labels = [], []
    for i, review in enumerate(train["review"]):
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=False))
        train_labels.append(train["sentiment"][i])

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=False))

    vocab = Vocab.build(clean_train_reviews, clean_test_reviews)

    train_reviews = [(vocab.convert_tokens_to_ids(sentence), train_labels[i])
                     for i, sentence in enumerate(clean_train_reviews)]
    test_reviews = [vocab.convert_tokens_to_ids(sentence)
                    for sentence in clean_test_reviews]

    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        train_reviews, train_labels, test_size=0.2, random_state=0
    )

    net = Transformer(vocab_size=len(vocab), embedding_dim=embed_size, hidden_dim=num_hiddens, num_class=labels)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = TransformerDataset(train_reviews)
    val_set = TransformerDataset(val_reviews)
    test_set = TransformerDataset(test_reviews)

    train_iter = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    # ================= 训练循环 ================= #
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, lengths, label in train_iter:
                n += 1
                net.zero_grad()
                feature = Variable(feature.to(device))
                lengths = Variable(lengths.to(device))
                label = Variable(label.to(device))
                score = net(feature, lengths)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
                train_loss += loss
                pbar.set_postfix({'epoch': '%d' % epoch,
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)})
                pbar.update(1)

            with torch.no_grad():
                for val_feature, val_length, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.to(device)
                    val_length = val_length.to(device)
                    val_label = val_label.to(device)
                    val_score = net(val_feature, val_length)
                    val_loss = loss_function(val_score, val_label)
                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % epoch,
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % runtime})
        torch.cuda.empty_cache()

    # ================= 预测阶段 ================= #
    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, lengths in test_iter:
                test_feature = test_feature.to(device)
                lengths = lengths.to(device)
                test_score = net(test_feature, lengths)
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    os.makedirs("./result", exist_ok=True)
    result_output.to_csv("./result/transformer.csv", index=False, quoting=3)
    logging.info('result saved!')
