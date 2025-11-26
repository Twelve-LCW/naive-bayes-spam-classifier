
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

class DataLoader:
    """
    统一加载预处理后的数据：
      - cleaned messages and labels
      - train/val/test indices
      - vocabulary
    所有路径相对于项目根目录。
    """

    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)

    def load_cleaned_data(self):
        """
        加载 cleaned_messages.csv
        Returns:
            messages (List[str]): 清洗后的邮件正文
            labels (List[int]): 对应标签 (0=ham, 1=spam)
        """
        csv_path = self.data_dir / 'cleaned_messages.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Cleaned data not found: {csv_path}")

        df = pd.read_csv(csv_path)
        messages = df['message'].tolist()
        labels = df['label'].tolist()
        return messages, labels

    def load_indices(self, split='train'):
        """
        加载指定划分的索引
        Args:
            split (str): 'train', 'val', or 'test'
        Returns:
            np.ndarray: 样本索引数组
        """
        valid_splits = {'train', 'val', 'test'}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}")

        index_file = self.data_dir / f'{split}_indices.npy'
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        return np.load(index_file)

    def load_vocab(self):
        """
        加载 vocab.json
        Returns:
            dict: 包含 'word_to_idx', 'idx_to_word' (可选), 'vocab_size', 'vocab'
        """
        vocab_file = self.data_dir / 'vocab.json'
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)

        # 可选：动态添加 idx_to_word（如果原始没有）
        if 'idx_to_word' not in vocab_dict:
            word_to_idx = vocab_dict['word_to_idx']
            idx_to_word = {idx: word for word, idx in word_to_idx.items()}
            vocab_dict['idx_to_word'] = idx_to_word

        return vocab_dict

    def get_split_data(self, split='train'):
        """
        获取指定划分的 (messages, labels) 列表
        Args:
            split (str): 'train', 'val', or 'test'
        Returns:
            messages_split (List[str])
            labels_split (List[int])
        """
        messages, labels = self.load_cleaned_data()
        indices = self.load_indices(split)
        messages_split = [messages[i] for i in indices]
        labels_split = [labels[i] for i in indices]
        return messages_split, labels_split