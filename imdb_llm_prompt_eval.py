# imdb_llm_prompt_eval.py
# 使用 llm_client 中的 LLMClient 对 IMDB 进行指令式情感分类：
# 1）在 labeledTrainData.tsv 上子集评估，输出 Val Acc
# 2）在 testData.tsv 上预测前 NUM_TEST 条，生成提交文件

import os
import time
import random
import pandas as pd
from tqdm import tqdm

from llm_client import LLMClient


# ====== 基本配置 ======
TRAIN_PATH = "corpus/imdb/labeledTrainData.tsv"
TEST_PATH = "corpus/imdb/testData.tsv"

# = -1 表示“用全部数据”，否则表示抽样数量
NUM_EVAL = 500       # 验证时随机抽多少条
NUM_TEST = 2000      # 测试集中预测多少条（25k 太慢，建议先 2000）

RESULT_DIR = "result"
RESULT_FILE = "llm_prompt_2000.csv"   # 最终结果文件名，按你以前命名风格来


def eval_on_labeled_data(client: LLMClient, train_path: str, num_eval: int) -> None:
    """
    在带标签的训练集上做子集评估，打印准确率和时间。
    """
    print("===== Step 1: Evaluate on labeledTrainData.tsv =====")
    df_train = pd.read_csv(train_path, sep="\t")
    print("Train head:")
    print(df_train.head())

    if num_eval == -1 or num_eval > len(df_train):
        num_eval = len(df_train)

    df_eval = df_train.sample(num_eval, random_state=42).reset_index(drop=True)

    correct = 0
    t0 = time.time()

    for i, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        text = row["review"]
        gold = int(row["sentiment"])

        pred = client.predict_imdb_label(text)
        if pred == gold:
            correct += 1

    elapsed = time.time() - t0
    acc = correct / len(df_eval)

    print(f"\nEval samples: {len(df_eval)}, Correct: {correct}, Accuracy: {acc:.4f}")
    print(f"Eval time: {elapsed/60:.1f} min  (~{elapsed/len(df_eval):.2f} s/样本)")

    # 方便你抄到表里：Val Acc ≈ ...
    print(f"\n[Summary] Val Acc ≈ {acc:.4f} （可以在表格中写成 {acc:.2f}）")


def predict_on_test_data(
    client: LLMClient,
    test_path: str,
    num_test: int,
    result_dir: str,
    result_file: str,
) -> None:
    """
    在无标签测试集上预测前 num_test 条，保存为 CSV。
    """
    print("\n===== Step 2: Predict on testData.tsv =====")
    df_test = pd.read_csv(test_path, sep="\t")
    print("Test head:")
    print(df_test.head())

    if num_test == -1 or num_test > len(df_test):
        num_test = len(df_test)

    preds = []
    t0 = time.time()

    for i in tqdm(range(num_test)):
        text = df_test.loc[i, "review"]
        pred = client.predict_imdb_label(text)
        preds.append(pred)

    elapsed = time.time() - t0
    print(f"\nPredicted {num_test} test samples, "
          f"time: {elapsed/60:.1f} min  (~{elapsed/num_test:.2f} s/样本)")

    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, result_file)

    result_df = pd.DataFrame({
        "id": df_test.loc[:num_test-1, "id"],
        "sentiment": preds,
    })
    result_df.to_csv(out_path, index=False)

    print(f"\n预测结果已保存到: {out_path}")
    print(result_df.head())


def main():
    # 固定随机种子，结果更可复现一点
    random.seed(42)

    # 1. 构建 LLM 客户端（模型搭建 & “API 接入”）
    client = LLMClient()

    # 2. 训练集子集评估（得到 Val Acc）
    eval_on_labeled_data(client, TRAIN_PATH, NUM_EVAL)

    # 3. 在测试集上预测，生成 result/llm_prompt_2000.csv
    predict_on_test_data(client, TEST_PATH, NUM_TEST, RESULT_DIR, RESULT_FILE)


if __name__ == "__main__":
    main()
