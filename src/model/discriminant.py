import numpy as np


class RDA:
    """
    Regularized Discrminant Analysis Model
    References:
        1. J. Friedman, 'Regularized discriminant anlaysis,'
           Taylor & Francis, 1989
        2. I. Pima & M. Aladjem, 'Regularized discriminant analysis for
           face recognintion,' Elsevier, 2004
        3. christopherjenness, 'https://github.com/christopherjenness/ML-lib/',
           2017
    """
    def __init__(
            self, alpha: float = 0.0, beta: float = 0.0
            ):
        """
        Args:
            alpha(float): regularization parameter (0 <= alpha <= 1)
            beta(float): additional regularization parameter (0 <= beta <= 1)
        """
        self.alpha = alpha
        self.beta = beta
        self.class_list = []
        self.class_prior = {}
        self.class_means = {}
        self.rda_cov = {}
        self.learn = False

    def reset(self):
        """
        Initialize model for re-training
        """
        self.class_list = []
        self.class_prior = {}
        self.class_means = {}
        self.rda_cov = {}
        self.learn = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X(np.ndarray): Train data(shape[n_samples, n_feautres])
            Y(np.ndarray): Class label(shape[n_samples, 1])
        Returns:
            Parameters of model is changed
        """
        self.class_list = np.unique(y)
        feature_num = x.shape[1]
        class_cov = {}
        reg_cov = {}
        pooled_cov = 0
        for c in self.class_list:
            x_index = np.where(y == c)[0]
            x_data = x[x_index, :]  # data of each class
            # prior probability
            self.class_prior[c] = float(len(x_index) / len(y))
            self.class_means[c] = x_data.mean(axis=0)  # mean vector
            class_cov[c] = np.cov(x_data, rowvar=0)  # covariance matrix
            # Calculate the pooled covariance estimate
            pooled_cov += class_cov[c] * self.class_prior[c]

        # Calculate regularized covariance matrics with lambda (here is alpha)
        for c in self.class_list:
            reg_cov[c] = \
                (((1 - self.alpha) * self.class_prior[c] * class_cov[c]) + self.alpha * pooled_cov) / \
                ((1 - self.alpha) * self.class_prior[c] + self.alpha)

        # Calcualte RDA covariance matrics with gamma (here is beta)
        for c in self.class_list:
            self.rda_cov[c] = \
                ((1 - self.beta) * reg_cov[c]) + \
                (self.beta / feature_num) * np.trace(reg_cov[c]) * np.eye(reg_cov[c].shape[0])
        self.learn = True

    def predict(self, x):
        """
        Args:
            X(np.ndarray): a row of data
        Returns:
            Pred(int): Prediction of label
        """
        # Model have to learn before prediction
        if self.learn is not True:
            raise ValueError("RDA model does not trained")

        # Prediction of distance
        class_dist = {}
        for c in self.class_list:
            diff_class = x - self.class_means[c]
            class_dist[c] = \
                np.matmul(np.matmul(diff_class.T, np.linalg.pinv(self.rda_cov[c])), diff_class)
            class_dist[c] += np.log(np.linalg.det(self.rda_cov[c]) + 1e-9)
            class_dist[c] -= 2 * np.log(self.class_prior[c])

        return min(class_dist, key=class_dist.get)
