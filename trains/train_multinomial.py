# train_multinomial.py

import os
from pathlib import Path
from utils.data_loader import DataLoader
from models.multinomial_nb import MultinomialNaiveBayes

def main():
    # 初始化数据加载器
    loader = DataLoader(data_dir='data')

    # 加载训练数据
    print("🔄 正在加载训练数据...")
    train_messages, train_labels = loader.get_split_data('train')

    # 加载词汇表
    print("🔄 正在加载词汇表...")
    vocab = loader.load_vocab()
    word_to_idx = vocab['word_to_idx']
    vocab_size = vocab['vocab_size']
    print(f"   词汇表大小: {vocab_size}")

    # 初始化并训练模型
    print("⚙️  开始训练 Multinomial Naive Bayes 模型...")
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(train_messages, train_labels, word_to_idx)

    # 保存模型
    output_dir = Path('saved_models')
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / 'multinomial_nb_model.pkl'

    model.save(model_path)

    print("\n🎉 训练完成！模型已保存，可用于后续预测或评估。")

if __name__ == "__main__":
    main()