from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from collections import Counter

# Fixed random seed to ensure reproducibility
RANDOM_SEED = 42
TRAIN_SIZE = 0.8
VAL_SIZE = 0.02
TEST_SIZE = 0.18  # Validation + Testing = 20%

def build_vocab_and_split(
    cleaned_csv='data/cleaned_messages.csv',  # Input path
    train_indices_path='data/train_indices.npy',
    val_indices_path='data/val_indices.npy',
    test_indices_path='data/test_indices.npy',
    vocab_path='data/vocab.json'
):
    """ 
    Split the cleaned_messages.csv file into training/validation/test sets 
    and construct a vocabulary (based only on the training set).
    """
    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Get the directory where the current script is located
    script_dir = Path(__file__).parent.parent

    # Build the full path
    cleaned_csv_full = os.path.join(script_dir, cleaned_csv)
    train_indices_full = os.path.join(script_dir, train_indices_path)
    val_indices_full = os.path.join(script_dir, val_indices_path)
    test_indices_full = os.path.join(script_dir, test_indices_path)
    vocab_full = os.path.join(script_dir, vocab_path)

    # Check if the input file exists.
    if not os.path.exists(cleaned_csv_full):
        raise FileNotFoundError(f"Input file not found: {cleaned_csv_full}")

    # Read the cleaned data
    df = pd.read_csv(cleaned_csv_full)
    messages = df['message'].tolist()
    labels = df['label'].tolist()
    n_samples = len(messages)

    # Create a full index
    all_indices = np.arange(n_samples)

    # Stratified sampling: divided by label
    spam_indices = [i for i, label in enumerate(labels) if label == 1]
    ham_indices = [i for i, label in enumerate(labels) if label == 0]

    # Reshuffle
    np.random.shuffle(spam_indices)
    np.random.shuffle(ham_indices)

    # Calculate the size of each set (proportional).
    n_train_spam = int(len(spam_indices) * TRAIN_SIZE)
    n_val_spam = int(len(spam_indices) * VAL_SIZE)
    n_test_spam = len(spam_indices) - n_train_spam - n_val_spam  # Preventing rounding errors

    n_train_ham = int(len(ham_indices) * TRAIN_SIZE)
    n_val_ham = int(len(ham_indices) * VAL_SIZE)
    n_test_ham = len(ham_indices) - n_train_ham - n_val_ham

    # Spam division
    train_spam = spam_indices[:n_train_spam]
    val_spam = spam_indices[n_train_spam : n_train_spam + n_val_spam]
    test_spam = spam_indices[n_train_spam + n_val_spam : n_train_spam + n_val_spam + n_test_spam]

    # Divide ham
    train_ham = ham_indices[:n_train_ham]
    val_ham = ham_indices[n_train_ham : n_train_ham + n_val_ham]
    test_ham = ham_indices[n_train_ham + n_val_ham : n_train_ham + n_val_ham + n_test_ham]

    # Merge and sort
    train_indices = np.sort(np.array(train_spam + train_ham))
    val_indices = np.sort(np.array(val_spam + val_ham))
    test_indices = np.sort(np.array(test_spam + test_ham))

    # Save Index
    os.makedirs(os.path.dirname(train_indices_full), exist_ok=True)
    np.save(train_indices_full, train_indices)
    np.save(val_indices_full, val_indices)
    np.save(test_indices_full, test_indices)

    print(f"Training set size: {len(train_indices)} ({len(train_indices)/n_samples:.1%})")
    print(f"Validation set size: {len(val_indices)} ({len(val_indices)/n_samples:.1%})")
    print(f"Test set size: {len(test_indices)} ({len(test_indices)/n_samples:.1%})")
    print(f"Indices saved to:")
    print(f"  - Train: {train_indices_full}")
    print(f"  - Val:   {val_indices_full}")
    print(f"  - Test:  {test_indices_full}")

    # === Constructing a vocabulary (based on the training set only)===
    word_counter = Counter()
    for idx in train_indices:
        msg = messages[idx]
        # Skip NaN, non-string, or empty messages
        if pd.isna(msg) or not isinstance(msg, str):
            continue
        msg = msg.strip()
        if not msg:
            continue
        words = msg.split()  # Cleaned, directly segmented into words
        word_counter.update(words)

    # Check if the vocabulary list is empty.
    if len(word_counter) == 0:
        raise ValueError("The vocabulary list is empty! Please check cleaned_messages.csv ")

    # Sorting ensures consistency of order.
    vocab = sorted(word_counter.keys())
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Save as JSON
    vocab_dict = {
        "word_to_idx": word_to_idx,
        "vocab_size": len(vocab),
        "vocab": vocab
    }
    with open(vocab_full, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print(f"The vocabulary list has been constructed! Vocabulary Size: {len(vocab)}")
    print(f"The vocabulary list has been saved to {vocab_full}")


if __name__ == "__main__":
    build_vocab_and_split()