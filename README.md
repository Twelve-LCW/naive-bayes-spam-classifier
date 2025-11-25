# Naive Bayes Spam Classifier
本项目实现 Multinomial NB 和 Bernoulli NB 分类器，用于垃圾邮件检测。

```markdown
naive-bayes-spam/
├── data/
│   ├── messages.csv                     # 原始数据集（未清洗，无 header）
│   ├── cleaned_messages.csv             # 清洗后的消息（仅 message + label）
│   ├── train_indices.npy                # 训练样本在 cleaned_messages.csv 中的行索引
│   ├── val_indices.npy                  # 验证样本索引
│   ├── test_indices.npy                 # 测试样本索引
│   └── vocab.json                       # 仅基于训练集构建的词汇表（word_to_idx 映射）
│
├── preprocessing/
│   ├── __init__.py
│   ├── text_processor.py                # 清洗文本 + 生成 cleaned_messages.csv
│   └── data_splitter.py                 # 划分训练/测试索引 + 构建 vocab.json
│
├── models/
│   ├── __init__.py
│   ├── multinomial_nb.py                # B: Multinomial Naive Bayes 实现
│   └── bernoulli_nb.py                  # A: Bernoulli Naive Bayes 实现
│
├── utils/
│   ├── data_loader.py                   # 统一加载 cleaned data + indices + vocab
│   └── model_io.py                      # 模型序列化保存/加载工具（如 pickle）
│
├── train/
│   ├── train_multinomial.py             # B训练脚本 → 输出 multinomial_model.pkl
│   └── train_bernoulli.py               # A训练脚本 → 输出 bernoulli_model.pkl
│
├── evaluate/
│   ├── predict_interface.py             # 统一预测接口（输入原始文本 → 输出 0/1）
│   └── evaluate_comparison.py           # 加载两个模型 + 测试集 → 输出 Precision/Recall 对比
│
├── saved_models/
│   ├── multinomial_model.pkl
│   └── bernoulli_model.pkl
│
├── README.md                            # 项目说明：如何运行、依赖、示例命令
└── report/                              # 实验报告、结果图表等
