
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

class DataLoader:
    """
    Load preprocessed data uniformly:
      - cleaned messages and labels
      - train/val/test indices
      - vocabulary
    """

    def __init__(self, data_dir='data'):
        # Get the parent directory (project root directory) of the directory where the current file is located.
        current_file = Path(__file__)
        root_dir = current_file.parent.parent  # utils -> project_root
        self.data_dir = root_dir / data_dir

    def load_cleaned_data(self):
        """
        Load cleaned_messages.csv
        Returns:
            messages (List[str]): Cleaned email body
            labels (List[int]): Corresponding tags (0=ham, 1=spam)
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
        Load the index of the specified partition
        Args:
            split (str): 'train', 'val', or 'test'
        Returns:
            np.ndarray: Sample index array
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
        Load vocab.json
        Returns:
            dict: Include 'word_to_idx', 'idx_to_word' (optinal), 'vocab_size', 'vocab'
        """
        vocab_file = self.data_dir / 'vocab.json'
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)

        if 'idx_to_word' not in vocab_dict:
            word_to_idx = vocab_dict['word_to_idx']
            idx_to_word = {idx: word for word, idx in word_to_idx.items()}
            vocab_dict['idx_to_word'] = idx_to_word

        return vocab_dict

    def get_split_data(self, split='train'):
        """
        Retrieves a list of (messages, labels) for a specified partition.
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