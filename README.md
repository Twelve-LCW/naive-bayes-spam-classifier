# Naive Bayes Spam Classifier

A simple yet effective spam email classifier based on the **Naive Bayes algorithm**, implemented in Python.  
This project includes two variants:
- **Multinomial Naive Bayes** (for word frequency-based features)
- **Bernoulli Naive Bayes** (for binary presence/absence of words)

The system supports training, evaluation, and prediction on new emails.

---

## 📁 Project Structure
```python
naive-bayes-spam-classifier/
│
├── data/                    # Raw and cleaned dataset
│   ├── messages.csv           # Original raw messages with labels
│   ├── cleaned_messages.csv   # Preprocessed text after cleaning
│   ├── vocab.json             # Vocabulary mapping (word → index)
│   └── *.npy                  # Train/val/test split indices
│
├── evaluate/                 # Evaluation and prediction scripts
│   ├── evaluate_multinomial.py    # Evaluate Multinomial NB on test set
│   ├── evaluate_bernoulli.py      # Evaluate Bernoulli NB on test set
│   ├── predict_multinomial.py     # Predict single email using Multinomial NB
│   └── predict_bernoulli.py       # Predict single email using Bernoulli NB
│
├── models/                   # Model implementations
│   ├── bernoulli_nb.py          # Bernoulli Naive Bayes classifier
│   └── multinomial_nb.py        # Multinomial Naive Bayes classifier
│
├── preprocessing/            # Data preprocessing utilities
│   ├── data_splitter.py         # Split data into train/val/test sets
│   └── text_processor.py        # Clean and tokenize text
│
├── saved_models/             # Trained model files (pickle format)
│   ├── bernoulli_nb_model_alpha*.pkl
│   └── multinomial_nb_model_alpha*.pkl
│
├── trains/                   # Training scripts
│   ├── train_bernoulli.py       # Train Bernoulli NB model
│   └── train_multinomial.py     # Train Multinomial NB model
│
├── utils/                    # Utility modules
│   └── data_loader.py           # Load data
│
├── .gitignore
└── README.md
```



## 🛠️ Environment Requirements

This project is tested and compatible with the following environment:

| Package | Version |
|--------|---------|
| Python | 3.9.23 |
| pandas | 2.2.3 |
| numpy  | 2.1.3 |
| json   | 2.0.9 |

> ✅ No external ML libraries (e.g., scikit-learn) are required — all metrics are computed manually using `numpy`.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Twelve-LCW/naive-bayes-spam-classifier.git
cd naive-bayes-spam-classifier
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install required packages via `requirements.txt` or manually:

```bash
pip install pandas==2.2.3 numpy==2.1.3
```

> Note: Standard `json` module is built-in; no installation needed.

---

## 🚀 Running the Program

### 📌 Prerequisites
Ensure that:
- The `data/cleaned_messages.csv`, `vocab.json`, and split indices exist.
- Models have been trained and saved to `saved_models/`.

If not, run training first.

---

### 🔹 Step 1: Train the Model

Run one of the training scripts to generate model files:

```bash
python trains/train_multinomial.py
python trains/train_bernoulli.py
```

> These scripts will save models like `multinomial_nb_model_alpha1.pkl` to `saved_models/`.

---

### 🔹 Step 2: Evaluate the Model

Evaluate performance on the test set:

```bash
python evaluate/evaluate_multinomial.py
python evaluate/evaluate_bernoulli.py
```

Output includes:
- Accuracy, Precision, Recall
- Confusion Matrix
- TP, FP, FN, TN counts
- Test set distribution

---

### 🔹 Step 3: Predict New Emails

Predict whether a given message is spam or ham:

```bash
python evaluate/predict_multinomial.py "Free money now! Click here!"
python evaluate/predict_bernoulli.py "Hi, how are you?"
```

Output:
```
Prediction: SPAM (1)
```

or

```
Prediction: HAM (0)
```

---

## 💡 Notes on Design Decisions

- **Text Processing**: Done once during preprocessing (`text_processor.py`) and stored as `cleaned_messages.csv`. All downstream steps use this cleaned data.
- **Model Loading**: Each evaluation/prediction script loads model and vocabulary independently for modularity.
- **No sklearn.metrics**: All evaluation metrics (accuracy, precision, recall) are computed manually using `numpy` for educational clarity and independence.
- **Modular Architecture**: Clear separation between preprocessing, training, evaluation, and prediction.

---

## 🖼️ Example Output

```text
=== Multinomial Naive Bayes Evaluation Results ===
Accuracy:  0.9962
Precision: 0.9778
Recall:    1.0000

Confusion Matrix:
                Predicted
                Ham (0)   Spam (1)
Actual Ham (0)     433        2
       Spam (1)      0        88

Basic Counts (Spam = Positive Class):
True Positives (TP):  88
False Positives (FP): 2
False Negatives (FN): 0
True Negatives (TN):  433
```

---

## IDE Support

This project works well with:
- **Visual Studio Code**
- **PyCharm**

✅ Recommended settings:
- Set project root as workspace
- Configure Python interpreter to point to your virtual environment

---

## 📚 Future Improvements

- Add support for TF-IDF weighting
- Implement cross-validation
- Integrate with web interface (Flask/FastAPI)
- Support multiple languages

---

## 📄 License

MIT License – feel free to use, modify, and distribute.

---

> © 2025 Luo Chengwei,Xu Jihao,Yang Hao. All rights reserved.