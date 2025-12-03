import os
from pathlib import Path
import sys
# Get the parent directory of the current file's directory (i.e., the project root directory).
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Points to the root directory of naive-bayes-spam-classifier
sys.path.append(str(project_root))

from utils.data_loader import DataLoader
from models.bernoulli_nb import BernoulliNaiveBayes

def main():
    # Initialize data loader
    loader = DataLoader(data_dir='data')

    # Load training data
    print("Training data is being loaded...")
    train_messages, train_labels = loader.get_split_data('train')

    # Load vocabulary list
    print("The vocabulary list is loading...")
    vocab = loader.load_vocab()
    word_to_idx = vocab['word_to_idx']
    vocab_size = vocab['vocab_size']
    print(f"Vocabulary list size {vocab_size}")

    # Initialize and train the model
    print("Start training the Bernoulli Naive Bayes model...")
    model = BernoulliNaiveBayes(alpha=0.1)
    model.fit(train_messages, train_labels, word_to_idx)

    # Save the model to saved_models/
    saved_models_dir = project_root / 'saved_models'
    saved_models_dir.mkdir(parents=True, exist_ok=True)  # Create directories (including parent directories). IF not exists
    model_path = saved_models_dir / 'bernoulli_nb_model_alpha0.1.pkl'

    model.save(model_path)

    print("\nTraining completed! The model has been saved.")

if __name__ == "__main__":
    main()