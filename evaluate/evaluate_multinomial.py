# evaluate/evaluate_multinomial.py

import sys
from pathlib import Path
import numpy as np

# Add project root to Python path to enable imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loader import DataLoader
from models.multinomial_nb import MultinomialNaiveBayes


def load_model_and_vocab():
    """
    Load the saved model and vocabulary.
    """
    model_path = project_root / 'saved_models' / 'multinomial_nb_model_alpha0.1.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    loader = DataLoader(data_dir=project_root / 'data')
    vocab = loader.load_vocab()
    word_to_idx = vocab['word_to_idx']

    model = MultinomialNaiveBayes.load(model_path, word_to_idx)
    return model, word_to_idx


def compute_metrics(y_true, y_pred):
    """
    Calculation of Accuracy, Precision, Recall, and Confusion Matrix
    Assume positive class is 1 (spam), negative class is 0 (ham)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    cm = np.array([[tn, fp],
                   [fn, tp]])  # shape: [[TN, FP], [FN, TP]]

    return accuracy, precision, recall, cm, tp, fp, fn, tn


def evaluate_on_test_set():
    """
    Evaluate the performance of the Multinomial Naive Bayes model on the test set.
    Output:Accuracy, precision, recall, confusion matrix, TP/FP/FN/TN
    """
    # Load model and vocabulary
    model, word_to_idx = load_model_and_vocab()

    # Load test data (raw messages and labels)
    loader = DataLoader(data_dir=project_root / 'data')
    test_messages, test_labels = loader.get_split_data('test')

    total_samples = len(test_labels)
    num_ham = sum(1 for label in test_labels if label == 0)
    num_spam = sum(1 for label in test_labels if label == 1)

    print(f"Loaded {total_samples} test samples.")
    print(f"Original test set distribution:")
    print(f"  Ham (label=0): {num_ham} ({num_ham / total_samples:.2%})")
    print(f"  Spam (label=1): {num_spam} ({num_spam / total_samples:.2%})")

    # Make predictions using the model's built-in predict method
    print("\nRunning predictions on test set...")
    predictions = model.predict(test_messages, word_to_idx)

    # Compute metrics without sklearn
    accuracy, precision, recall, cm, tp, fp, fn, tn = compute_metrics(test_labels, predictions)

    # Output results
    print("\n=== Multinomial Naive Bayes Evaluation Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Ham (0)   Spam (1)")
    print(f"Actual Ham (0)   {cm[0, 0]:>6}     {cm[0, 1]:>6}")
    print(f"       Spam (1)  {cm[1, 0]:>6}     {cm[1, 1]:>6}")
    print("\nBasic Counts (Spam = Positive Class):")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN):  {tn}")
    print("==================================================")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'test_distribution': {'ham': num_ham, 'spam': num_spam, 'total': total_samples}
    }


if __name__ == "__main__":
    evaluate_on_test_set()