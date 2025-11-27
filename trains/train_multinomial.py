# train_multinomial.py

import os
from pathlib import Path
import sys
# 获取当前文件所在目录的父目录（即项目根目录）
current_dir = Path(__file__).parent
project_root = current_dir.parent  # 指向 naive-bayes-spam-classifier 根目录
sys.path.append(str(project_root))

from utils.data_loader import DataLoader
from models.multinomial_nb import MultinomialNaiveBayes

def main():
    # 初始化数据加载器
    loader = DataLoader(data_dir='data')

    # 加载训练数据
    print("Training data is being loaded...")
    train_messages, train_labels = loader.get_split_data('train')

    # 加载词汇表
    print("The vocabulary list is loading...")
    vocab = loader.load_vocab()
    word_to_idx = vocab['word_to_idx']
    vocab_size = vocab['vocab_size']
    print(f"Vocabulary list size {vocab_size}")

    # 初始化并训练模型
    print("Start training the Multinomial Naive Bayes model...")
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(train_messages, train_labels, word_to_idx)

    # 保存模型到 saved_models/
    saved_models_dir = project_root / 'saved_models'
    saved_models_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（包括父目录）
    model_path = saved_models_dir / 'multinomial_nb_model_alpha0.5.pkl'

    model.save(model_path)

    print("\nTraining completed! The model has been saved.")

if __name__ == "__main__":
    main()