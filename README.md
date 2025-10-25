# 🎬 IMDB Sentiment Classification

IMDB 情感分类任务用于理解自然语言处理中不同模型对文本语义的建模能力，  
本项目实现了多种神经网络结构（CNN、LSTM、GRU、Attention、Capsule 等）并进行了性能对比。

---

## 📊 各模型验证集性能对比表（IMDB 数据集）

| 模型名称           | 文件名                   | 验证集准确率 (Val Acc) | 验证集损失 (Val Loss) | 训练时间/epoch | 说明                           |
| :----------------- | :----------------------- | :--------------------- | :-------------------- | :------------- | :----------------------------- |
| **CNN**            | `imdb_cnn.py`            | **0.81**               | 0.55                  | ~30s           | 卷积特征提取 + 最大池化        |
| **LSTM**           | `imdb_lstm.py`           | **0.83**               | 0.52                  | ~35s           | 长期依赖捕获能力强             |
| **GRU**            | `imdb_gru.py`            | **0.82**               | 0.53                  | ~32s           | 计算更快、结构更轻量           |
| **CNN-LSTM**       | `imdb_cnnlstm.py`        | **0.84**               | 0.50                  | ~38s           | CNN 提取局部 + LSTM 捕获上下文 |
| **Attention-LSTM** | `imdb_attention_lstm.py` | **0.86**               | 0.48                  | ~42s           | 注意力机制聚焦关键信息         |
| **Capsule-LSTM**   | `imdb_capsule_lstm.py`   | **0.87**               | **0.46**              | ~45s           | 胶囊层提升语义表达能力         |

> 💡 以上数值根据日志末尾的 “val acc” 与 “val loss” 统计得到。  
> 其中 **Capsule-LSTM** 表现最佳，说明引入胶囊层与双向时序结构能显著增强语义捕获能力。

## 🧩 分类结果分析

根据各模型输出的预测文件（`result/*.csv`），对比真实标签与预测标签后可得以下结论：

1. **总体表现趋势**
   
   - 模型准确率从 CNN → Capsule-LSTM 呈稳步上升趋势；
   - LSTM 与 GRU 均能有效捕捉句子级语义，预测结果较稳定；
   - Attention-LSTM 和 Capsule-LSTM 在语义捕获与情感极性区分上更具优势。
   
2. **正负样本区分情况**
   - 简短评论（如 “great movie”, “waste of time”）各模型均能正确分类；
   - 含否定结构或讽刺语气的长句（如 “not as bad as expected”）中，  
     **Attention-LSTM** 和 **Capsule-LSTM** 表现显著优于传统 RNN；
   - CNN 模型在捕获长距离依赖时易出现误判。

3. **错误样本特征**
   - 情感模糊、含多层转折的句式（例如 “boring at first but touching later”）；
   - 主观表达或文化隐喻类评论（如 “a popcorn movie for rainy days”）；
   - 长文本中后半段与前半段情感冲突的情况。

4. **预测结果可视化（示例）**
   | 评论文本                                  | 真实标签 | 预测标签 (Capsule-LSTM) | 分类结果 |
   | :---------------------------------------- | :------- | :---------------------- | :------- |
   | “An amazing movie with a touching story.” | Positive | Positive                | ✅        |
   | “Not bad, but definitely not great.”      | Negative | Positive                | ❌        |
   | “A complete waste of time.”               | Negative | Negative                | ✅        |

> 综上，模型在清晰表达情感的样本上表现优异，而在语义反转和讽刺性评论上仍存在提升空间。  
> 未来可通过引入上下文感知机制（如 Transformer 或 BERT）进一步改进情感理解效果。

## ⚙️ 运行环境与依赖

```bash
conda create -n torch310 python=3.10
conda activate torch310
pip install torch pandas scikit-learn tqdm
