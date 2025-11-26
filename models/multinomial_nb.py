import numpy as np
from collections import defaultdict
from math import log

class MultinomialNaiveBayes:
    """
     Multinomial Naive Bayes 分类器（用于垃圾邮件检测）
    - 支持拉普拉斯平滑
    - 假设输入为词频向量（基于共享 vocab)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.pi_spam = 0.0      # P(spam)
        self.pi_ham = 0.0       # P(ham)
        self.theta_spam = None  # P(word_j | spam)，长度 = vocab_size
        self.theta_ham = None   # P(word_j | ham)
        self.vocab_size = 0
        self.is_fitted = False

    def _vectorize(self, messages, word_to_idx):
        """
        将消息列表转换为词频矩阵(list of lists)
        Args:
            messages: List[str]，每条是清洗后的文本（如 "hello world hello")
            word_to_idx: Dict[str, int]
        Returns:
            X: List[List[int]]，每个元素是长度为 vocab_size 的词频向量
        """
        X = []
        for msg in messages:
            freq = [0] * self.vocab_size
            words = msg.split()
            for w in words:
                if w in word_to_idx:
                    idx = word_to_idx[w]
                    freq[idx] += 1
            X.append(freq)
        return X

    def fit(self, train_messages, train_labels, word_to_idx):
        """
        训练 Multinomial NB 模型
        Args:
            train_messages: List[str] — 清洗后的训练邮件正文
            train_labels: List[int] — 对应标签 (0=ham, 1=spam)
            word_to_idx: Dict[str, int] — 共享词汇表映射
        """
        self.vocab_size = len(word_to_idx)
        word_to_idx = word_to_idx  # 本地引用

        # 向量化训练数据
        X_train = self._vectorize(train_messages, word_to_idx)
        y_train = train_labels

        # 统计类别数量
        N_spam = sum(1 for y in y_train if y == 1)
        N_ham = len(y_train) - N_spam
        N_total = len(y_train)

        # 计算先验概率（带极小值防止 log(0)，但通常不会为0）
        self.pi_spam = N_spam / N_total
        self.pi_ham = N_ham / N_total

        # 初始化词频统计
        count_spam = [0] * self.vocab_size  # L_spam,j
        count_ham = [0] * self.vocab_size   # L_ham,j
        total_words_spam = 0
        total_words_ham = 0

        # 遍历训练样本，累加词频
        for x, y in zip(X_train, y_train):
            if y == 1:  # spam
                for j, freq in enumerate(x):
                    if freq > 0:
                        count_spam[j] += freq
                        total_words_spam += freq
            else:  # ham
                for j, freq in enumerate(x):
                    if freq > 0:
                        count_ham[j] += freq
                        total_words_ham += freq

        # 拉普拉斯平滑：θ_cj = (L_cj + α) / (L_c + α * V)
        V = self.vocab_size
        self.theta_spam = [
            (count_spam[j] + self.alpha) / (total_words_spam + self.alpha * V)
            for j in range(V)
        ]
        self.theta_ham = [
            (count_ham[j] + self.alpha) / (total_words_ham + self.alpha * V)
            for j in range(V)
        ]

        self.is_fitted = True
        print("Multinomial NB 模型训练完成！")
        print(f"   Spam prior: {self.pi_spam:.4f}, Ham prior: {self.pi_ham:.4f}")
        print(f"   Vocabulary size: {self.vocab_size}")

    def predict_log_proba(self, messages, word_to_idx):
        """
        （内部方法）返回 log P(spam|x) 和 log P(ham|x) 的未归一化得分
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = self._vectorize(messages, word_to_idx)
        log_probs_spam = []
        log_probs_ham = []

        log_pi_spam = log(self.pi_spam)
        log_pi_ham = log(self.pi_ham)

        for x in X:
            # 计算 log P(x|spam) + log P(spam)
            score_spam = log_pi_spam
            score_ham = log_pi_ham

            for j, freq in enumerate(x):
                if freq > 0:
                    # 避免 log(0)，但理论上不会发生（因拉普拉斯平滑）
                    score_spam += freq * log(self.theta_spam[j])
                    score_ham += freq * log(self.theta_ham[j])

            log_probs_spam.append(score_spam)
            log_probs_ham.append(score_ham)

        return log_probs_spam, log_probs_ham

    def predict(self, messages, word_to_idx):
        """
        预测标签
        Args:
            messages: List[str]
            word_to_idx: Dict[str, int]
        Returns:
            predictions: List[int] (0 or 1)
        """
        log_spam, log_ham = self.predict_log_proba(messages, word_to_idx)
        return [1 if s > h else 0 for s, h in zip(log_spam, log_ham)]
    # ===== 添加到 MultinomialNaiveBayes 类中 =====

    def save(self, filepath):
        """
        保存模型所有必要参数到文件（使用 pickle）
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
        print(f"模型已保存至: {filepath}")

    @classmethod
    def load(cls, filepath, word_to_idx):
        """
        从文件加载模型，并绑定词汇表映射
        Args:
            filepath: 模型文件路径
            word_to_idx: Dict[str, int] — 必须与训练时一致
        Returns:
            实例化的 MultinomialNaiveBayes 对象
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

        # 注意：word_to_idx 不保存在模型中（由外部提供），但必须一致
        model._external_word_to_idx = word_to_idx  # 仅用于 predict 接口兼容性（可选）

        return model