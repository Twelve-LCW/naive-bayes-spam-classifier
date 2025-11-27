# evaluate/predict_multinomial.py

import re
import sys
from pathlib import Path

# === 关键修复：动态将项目根目录加入 Python 路径 ===
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ==============================================

from utils.data_loader import DataLoader
from models.multinomial_nb import MultinomialNaiveBayes


def load_model_and_vocab():
    model_file = project_root / 'saved_models' / 'multinomial_nb_model.pkl'
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    loader = DataLoader(data_dir=project_root / 'data')
    vocab = loader.load_vocab()
    word_to_idx = vocab['word_to_idx']

    model = MultinomialNaiveBayes.load(model_file, word_to_idx)
    return model, word_to_idx


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if 2 <= len(w) <= 20]
    return ' '.join(words)


def predict_spam_multinomial(text: str) -> int:
    try:
        model, word_to_idx = load_model_and_vocab()
        cleaned = clean_text(text) or 'aaa'
        return model.predict([cleaned], word_to_idx)[0]
    except Exception as e:
        print(f"[Error] Prediction failed: {e}", file=sys.stderr)
        return -1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_multinomial.py \"Your email content here\"")
        sys.exit(1)

    result = predict_spam_multinomial(sys.argv[1])
    label = "SPAM" if result == 1 else "HAM" if result == 0 else "ERROR"
    print(f"Prediction: {label} ({result})")
    if result == -1:
        sys.exit(1)