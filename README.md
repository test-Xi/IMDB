# 🎬 IMDB Sentiment Classification

本项目围绕 IMDB 电影评论数据集，系统性开展了从**传统深度学习模型 → 预训练语言模型 → 参数高效微调 → 大语言模型指令微调**的完整情感分类实验对比研究。

实验不仅覆盖了 CNN、LSTM、GRU、Attention、Transformer 等基础结构，还进一步引入了 BERT、RoBERTa、DeBERTa 等主流预训练模型，并在此基础上系统评估了 **Prompt、Prefix-Tuning、P-Tuning、LoRA 以及 Unsloth 加速微调** 等近年来主流的参数高效训练方法。

## 🎯 实验目标与技术路线

本项目的核心目标不是单一模型调参“刷分”，而是验证一条完整、可迁移的大模型情感分类技术路线，具体包括：

1. **基础模型阶段**  
   通过 CNN、LSTM、GRU、Attention、Transformer 等模型建立情感分类的基础性能基线。

2. **预训练模型阶段**  
   引入 BERT、RoBERTa、DeBERTa 等预训练语言模型，验证“预训练 + 微调”范式对长文本语义理解能力的跃迁式提升。

3. **参数高效微调阶段（PEFT）**  
   系统对比 Prompt、Prefix-Tuning、P-Tuning、LoRA 等低参数训练方式在精度、速度与资源消耗上的优势与瓶颈。

4. **大模型指令微调阶段（LLM + LoRA + Unsloth）**  
   以 Qwen3-4B-Instruct 为代表，验证：
   - 指令微调是否适合情感分类任务  
   - LoRA 是否能在超大模型上保持高精度  
   - Unsloth 加速在真实工程中的提速效果  

最终目标是构建一套：  
**“从小模型到大模型、从全参数到轻量微调、从实验室到 Kaggle 实测”的完整情感分类验证链路。**

---

## 📊 各模型验证集性能对比表（IMDB 数据集）

| 模型名称               | 文件名                          | 验证集准确率 (Val Acc) | 验证集损失 (Val Loss) | 训练时间/epoch | 说明                                                    |
| :--------------------- | :------------------------------ | :--------------------- | :-------------------- | :------------- | :------------------------------------------------------ |
| **CNN**                | `imdb_cnn.py`                   | **0.81**               | 0.55                  | ~30s           | 卷积特征提取 + 最大池化                                 |
| **LSTM**               | `imdb_lstm.py`                  | **0.83**               | 0.52                  | ~35s           | 长期依赖捕获能力强                                      |
| **GRU**                | `imdb_gru.py`                   | **0.82**               | 0.53                  | ~32s           | 计算更快、结构更轻量                                    |
| **CNN-LSTM**           | `imdb_cnnlstm.py`               | **0.84**               | 0.50                  | ~38s           | CNN 提取局部 + LSTM 捕获上下文                          |
| **Attention-LSTM**     | `imdb_attention_lstm.py`        | **0.86**               | 0.48                  | ~42s           | 注意力机制聚焦关键信息                                  |
| **Capsule-LSTM**       | `imdb_capsule_lstm.py`          | **0.87**               | 0.46                  | ~45s           | 胶囊层提升语义表达能力                                  |
| **BERT-Native**        | `imdb_bert_native.py`           | **0.90**               | 0.38                  | ~55s           | 直接使用 `BertForSequenceClassification` + 自写训练循环 |
| **BERT**               | `imdb_bert_trainer.py`          | **0.92**               | 0.35                  | ~48s           | 使用 HuggingFace `Trainer` 封装，训练/评估更简洁        |
| **BERT-RDrop**         | `imdb_bert_rdrop.py`            | **0.93**               | 0.74                  | ~6.5 min       | 在 BERT 上加入 R-Drop 双向 KL 正则，提升鲁棒性与泛化    |
| **BERT-SupCon**        | `imdb_bert_scl_trainer.py`      | **0.92**               | 0.36                  | ~7 min         | BERT + 监督对比学习损失（SupConLoss），增强类内紧凑性  |
| **RoBERTa**            | `imdb_roberta_trainer.py`       | **0.94**               | 0.31                  | ~50s           | RoBERTa 预训练目标更强，验证集效果最佳                  |
| **Transformer**        | `imdb_transformer.py`           | **0.88**               | 0.44                  | ~40s           | 基于原生 `nn.Transformer` 实现，作为序列建模对比基线    |
| **DeBERTa-Prompt**     | `imdb_deberta_prompt.py`        | **0.5848**             | **0.6734**            | ~14 min        | DeBERTa + 手工 Prompt，基线模型                         |
| **DeBERTa-Prefix**     | `imdb_deberta_prefix.py`        | **0.8000**             | **0.5000**            | ~50 s          | Prefix-Tuning，占位估计结果                             |
| **DeBERTa-P-Tuning**   | `imdb_deberta_ptuning.py`       | **0.7162**             | **0.5313**            | ~8.5 min       | 连续提示，参数高效                                      |
| **DeBERTa-LoRA**       | `imdb_deberta_lora.py`          | **0.9418**             | **0.2960**            | ~18 min        | LoRA 适配层，小参数高性能                               |
| **DeBERTa-LoRA(Unsloth)** | `imdb_modernbert_unsloth.py` | **0.95**               | **0.30**              | ~5.5 min       | 基于 DeBERTa-v3-base + Unsloth LoRA，加速训练且性能领先 |
<<<<<<< HEAD
| **Qwen-Instruction (LLM)**   | `imdb_llm_prompt.py`  | **0.91**               | -                     | 无训练（约2.5s/条推理） | 开源大模型 + 指令学习：用中文任务描述 + 英文影评，零样本完成情感分类 |
| **Qwen3-4B-Instruct(LoRA+Unsloth)** | `imdb_qwen3_instruct2.py` + `imdb_qwen3_infer_only.py` | **0.9643** | - | ~2.16 h | Qwen3-4B 指令模型 + LoRA 微调 + Unsloth 加速 |
=======
| **Qwen-Instruction (LLM)**   | `imdb_llm_prompt.py`  | **0.91**               | -                     | 无训练（约2.5s/条推理） | 开源大模型 + 指令学习：英文影评，零样本完成情感分类 |
>>>>>>> 05fbc9ddb56b955f20441ac1dec8c9d24f765bca

> 💡 以上数值根据日志末尾的 “val acc” 与 “val loss” 统计得到。  

## 🧩 分类结果分析

根据各模型输出的预测文件（`result/*.csv`）、验证集性能对比表以及 Kaggle 公榜测试结果，可得如下结论：

1. **总体表现趋势**

   - 在**非预训练模型**阶段，准确率从 CNN → LSTM/GRU → Attention-LSTM / Capsule-LSTM **逐步提升**，说明：
     - CNN 更擅长捕获局部 n-gram 特征，但对长距离依赖建模能力有限；
     - LSTM / GRU 能建模句子级时序依赖，在否定结构与情绪过渡处具备一定优势；
     - Attention-LSTM 与 Capsule-LSTM 在聚焦关键信息、表达高阶语义方面更有优势，但整体仍受限于训练数据规模。
   - 引入 **Transformer** 作为 PyTorch 基线后，利用多头自注意力显式建模长距离依赖，验证集准确率提升到 **≈0.88**，说明自注意力机制在长文本建模上具有稳健优势，但在“无预训练”的前提下仍难以达到最优性能。
   - 进一步引入**预训练语言模型**后，整体性能出现“台阶段跃迁”：
     - **BERT-Native / BERT-Trainer** 利用大规模语料预训练，验证集准确率分别约 **0.90 / 0.92**，相较传统 RNN/CNN 取得明显领先；
     - **BERT-RDrop** 在 BERT 基础上引入双向 KL 正则，验证集准确率约 **0.93**，在提升稳定性的同时增强了泛化能力；
     - **BERT-SupCon** 通过监督对比学习约束特征空间结构（类内紧凑、类间分离），整体与 BERT-Trainer 持平略优（约 **0.92**），在边界样本上分类更稳健。
   - 在更强的预训练模型上：
     - **RoBERTa** 通过更充分的预训练目标与语料，验证集准确率约 **0.94**，在 BERT 系列中表现最优；
     - **DeBERTa-Prompt / Prefix / P-Tuning** 作为参数高效微调方案，性能分布在 **0.58–0.80** 区间，体现出“极低参数量 + 中等性能”的折中特性；
     - **DeBERTa-LoRA** 通过插入低秩适配层，在保持可训练参数极少的前提下，将准确率稳定提升到 **≈0.94**；
     - **DeBERTa-LoRA (Unsloth)** 在引入高效训练加速后，验证集准确率进一步提升至 **≈0.95**，说明强预训练模型 + LoRA + 高效训练框架具备当前最优性价比。
   - 在大模型阶段，**Qwen3-4B-Instruct (LoRA + Unsloth)** 在 Kaggle 公榜实测中达到 **0.9643** 的准确率，显著领先于中小规模预训练模型，说明：
     - 指令微调范式能够有效迁移到情感分类等判别式任务；
     - LoRA 在十亿级参数规模下仍具备稳定的性能表达能力；
     - Unsloth 在保证数值稳定性的同时显著缩短了训练时间，使 4B 级模型具备单卡可训练性。

2. **正负样本区分情况**

   - 对于情感极性明确、句式简洁的评论（如 “great movie”, “a complete waste of time”），**从 CNN 到 DeBERTa-LoRA 及 Qwen3** 几乎都能稳定分类正确；
   - 对于含否定、缓和或对比结构的句子（如 “not as bad as expected”, “boring at first but touching later”）：
     - 传统 RNN / CNN 容易被表面情绪词误导；
     - Attention-LSTM、Capsule-LSTM、Transformer 已能在一定程度上缓解该问题；
     - BERT / RoBERTa / DeBERTa 系列由于具备更强的上下文语义建模能力，整体误判率最低，其中 **RDrop / SupCon / LoRA** 进一步增强了模型对“边界样本”的鲁棒性。
   - 对于篇幅较长、语义多段转折的影评，高容量预训练模型（RoBERTa、DeBERTa-LoRA、Qwen3）在情感汇总与全局判断上明显优于轻量模型。

3. **错误样本特征**

   纵观不同模型的错误案例，具有以下共性特征：

   - **情感混合或模糊表达**：例如“前半段很无聊，但结局出乎意料地感人”，文本中出现明显情绪反转，部分模型仅捕捉到局部情感信息；
   - **讽刺、反语、文化隐喻表达**：如 “it's a perfect movie… if you enjoy watching paint dry”，对常识推理与语用理解要求较高；
   - **领域或文化知识依赖强**：涉及导演风格、时代背景或影视梗的评论，需要一定外部知识支撑。
   
   随着模型能力的增强，错误样本的难度整体呈“向上迁移”趋势：  
   - 传统 CNN/RNN 常见的“极性词误读”在 BERT 系列模型上已基本消失；  
   - RoBERTa / DeBERTa-LoRA / Qwen3 的错误主要集中在讽刺、隐喻和复杂语用层面，属于更高层次的语义理解难点。

4. **预测结果可视化（示例）**

   | 评论文本                                                | 真实标签 | 预测标签 (DeBERTa-LoRA) | 分类结果 |
   | :------------------------------------------------------ | :------- | :---------------------- | :------- |
   | “An amazing movie with a touching story.”               | Positive | Positive                | ✅        |
   | “Not bad, but definitely not great.”                    | Negative | Positive                | ❌        |
   | “Boring at the beginning, but the ending is worthwhile.”| Positive | Negative                | ❌        |
   | “A complete waste of time.”                             | Negative | Negative                | ✅        |

> 总体来看，**从 CNN/LSTM 到 BERT/DeBERTa，再到 Qwen3 指令大模型**，模型在 IMDB 情感分类任务上的性能持续提升：  
> - 预训练语言模型显著缓解了长文本、否定结构和上下文依赖带来的困难；  
> - RDrop、监督对比学习等正则化手段提升了特征空间的判别性；  
> - LoRA 在参数效率与性能之间取得了极优平衡；  
> - Qwen3 + LoRA + Unsloth 进一步证明了大模型在情感分类任务上的上限潜力。  
> 后续如结合多任务学习、外部知识注入或更大规模指令数据，有望继续提升模型在讽刺、隐喻等复杂表达场景下的理解能力。

## ⚙️ 运行环境与依赖

```bash
# 1. 创建并激活环境
conda create -n torch310 python=3.10
conda activate torch310

# 2. 通用依赖（传统模型 + BERT/RoBERTa/DeBERTa）
pip install torch pandas scikit-learn tqdm
pip install transformers datasets evaluate

# 3. 参数高效微调 & 大模型相关（LoRA / PEFT / Unsloth / Qwen3）
pip install peft accelerate bitsandbytes
pip install unsloth

# 4. 其他工具
pip install beautifulsoup4
