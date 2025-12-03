from math import log


class BernoulliNaiveBayes:
    """
     Bernoulli Naive Bayes Classifier
    -Input:Word existence vectors(vocab.json)
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.pi_spam = 0.0  # P(spam)
        self.pi_ham = 0.0  # P(ham)
        self.theta_spam = None  # P(word_j | spam)，len = vocab_size
        self.theta_ham = None  # P(word_j | ham)
        self.vocab_size = 0
        self.is_fitted = False

    def _vectorize(self, messages, word_to_idx):
        """
        Convert the message list into a word existence matrix
        Args:
            messages: List[str],Each line is the cleaned text.(For example "hello world hello")
            word_to_idx: Dict[str, int]
        Output:
            X: List[List[int]],Each element is a word existence vector of length vocab_size.
        """
        X = []
        for msg in messages:
            exist = [0] * self.vocab_size
            words = msg.split()
            for w in words:
                if w in word_to_idx:
                    idx = word_to_idx[w]
                    exist[idx] = 1
            X.append(exist)

        return X

    def fit(self, train_messages, train_labels, word_to_idx):
        """
        train Bernoulli NB model
        Args:
            train_messages: List[str] — Cleaned training email body
            train_labels: List[int] — Labels(0=ham, 1=spam)
            word_to_idx: Dict[str, int] — Shared vocabulary mapping
        """
        self.vocab_size = len(word_to_idx)
        X_train = self._vectorize(train_messages, word_to_idx)
        y_train = train_labels

        N_spam = sum(1 for y in y_train if y == 1)
        N_ham = len(y_train) - N_spam
        N_total = len(y_train)

        # calculate prior
        self.pi_spam = N_spam / N_total
        self.pi_ham = N_ham / N_total

        exist_spam = [0] * self.vocab_size
        exist_ham = [0] * self.vocab_size

        for x, y in zip(X_train, y_train):
            if y == 1:
                exist_spam = [a + b for a, b in zip(exist_spam, x)]
            else:
                exist_ham = [a + b for a, b in zip(exist_ham, x)]

        # Laplace smooth: θ_cj = (N_cj + α) / (N_c + α * V)
        self.theta_spam = [
            (exist_spam[j] + self.alpha) / (N_spam + self.alpha * 2)
            for j in range(self.vocab_size)
        ]
        self.theta_ham = [
            (exist_ham[j] + self.alpha) / (N_ham + self.alpha * 2)
            for j in range(self.vocab_size)
        ]

        self.is_fitted = True
        print("Bernoulli NB model training is complete!")
        print(f"   Spam prior: {self.pi_spam:.4f}, Ham prior: {self.pi_ham:.4f}")
        print(f"   Vocabulary size: {self.vocab_size}")

    def predict_log_proba(self, messages, word_to_idx):
        """
        (Internal method) Returns the unnormalized scores of log P(spam|x) and log P(ham|x).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = self._vectorize(messages, word_to_idx)
        log_probs_spam = []
        log_probs_ham = []

        log_pi_spam = log(self.pi_spam)
        log_pi_ham = log(self.pi_ham)

        for x in X:
            # calculate log P(x|spam) + log P(spam)
            score_spam = log_pi_spam
            score_ham = log_pi_ham
            for j, isExist in enumerate(x):
                score_spam += log(self.theta_spam[j]) if isExist else log(1 - self.theta_spam[j])
                score_ham += log(self.theta_ham[j]) if isExist else log(1 - self.theta_ham[j])

            log_probs_spam.append(score_spam)
            log_probs_ham.append(score_ham)

        return log_probs_spam, log_probs_ham

    def predict(self, messages, word_to_idx):
        """
        Predicted Labels
        Args:
            messages: List[str]
            word_to_idx: Dict[str, int]
        Returns:
            predictions: List[int] (0 or 1)
        """
        log_spam, log_ham = self.predict_log_proba(messages, word_to_idx)
        return [1 if s > h else 0 for s, h in zip(log_spam, log_ham)]

    def save(self, filepath):
        """
        Save all necessary model parameters to a file (pickle).
        """
        import pickle
        model_data = {
            'pi_spam': self.pi_spam,
            'pi_ham': self.pi_ham,
            'theta_spam': self.theta_spam,
            'theta_ham': self.theta_ham,
            'vocab_size': self.vocab_size,
            'alpha': self.alpha,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"The model has been saved to: {filepath}")

    @classmethod
    def load(cls, filepath, word_to_idx):
        """
        Load the model from the file and bind the vocabulary mapping.
        Args:
            filepath: Model file path
            word_to_idx: Dict[str, int] — Must be consistent with training
        Returns:
            Instantiated MultinomialNaiveBayes objects
        """
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(alpha=model_data.get('alpha', 1.0))
        model.pi_spam = model_data['pi_spam']
        model.pi_ham = model_data['pi_ham']
        model.theta_spam = model_data['theta_spam']
        model.theta_ham = model_data['theta_ham']
        model.vocab_size = model_data['vocab_size']
        model.is_fitted = model_data['is_fitted']

        # The word_to_idx is not stored in the model (it is provided externally), but it must be consistent.
        model._external_word_to_idx = word_to_idx  # For predict interface compatibility only

        return model