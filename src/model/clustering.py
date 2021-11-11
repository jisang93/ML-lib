import numpy as np

from scipy.stats import multivariate_normal


# Gaussian Mixture Model
class GaussianMixture:
    """
    Gaussian Mixture Model for clustering
    """
    def __init__(self, k: int):
        """
        Args:
            k(int): the number of cluster
        Attribute
            X(np.ndarray): data for clustering
            menas(np.ndarray): means of classes
            covs(np.ndarray): covariance of classes
            weights(np.ndarray): prior probability of classes
            responsibility(np.ndarray): responsibility matrix in EM algorithms
            convergence(boolean): check model has been converged
        """
        self.X = np.nan
        self.k = k
        self.means = np.zeros(k)
        self.covs = np.nan
        self.weights = np.nan
        self.responsibility = np.nan
        self.loglikelihood = 0
        self.convergence = False

    def fit(self, X: np.ndarray) -> None:
        """
        Args:
            data(np.ndarray): training data
        Returns:
            None. Variables changed.
        """
        # Check training data
        self.X = X
        num_x = np.shape(X)[0]
        num_feature = np.shape(X)[1]
        # Initialize for clustering
        init_indexes = \
            np.random.choice(range(num_x), self.k, replace=False)
        # Initial settings
        self.means = X[init_indexes, :]
        self.covs = [np.identity(num_feature) for _ in range(self.k)]
        self.weights = [1.0 / self.k for _ in range(self.k)]
        
        # Fitting
        likelihood = []
        while True:
            # EM algorithm
            self.responsibility = self._expectation()
            self.weights, self.means, self.covs = self._maximization()
            # Calculate log-likelihood
            new_loglikelihood = np.sum(np.apply_along_axis(self._log_likelihood, 1, self.X))
            likelihood.append(new_loglikelihood)
            # Break condition
            if np.abs(self.loglikelihood - new_loglikelihood) <= 1e-6:
                break
            else:
                self.loglikelihood = new_loglikelihood
        # Convergence
        self.convergence = True

        return likelihood

    def _expect(self, x):
        # Expectation step
        responsibility = np.zeros(self.k)
        for c in range(self.k):
            responsibility[c] = \
                multivariate_normal.pdf(
                                     x,
                                     mean=self.means[c],
                                     cov=self.covs[c]
                                     ) * self.weights[c]
        responsibility = \
            [float(prob) / sum(responsibility) for prob in responsibility]

        return responsibility

    def _expectation(self):
        return np.apply_along_axis(self._expect, 1, self.X)
    
    def _maximization(self):
        # Initialize variance
        weights = np.zeros_like(self.weights)
        means = np.zeros_like(self.means)
        covs = np.zeros_like(self.covs)
        # Maximaization porcess
        priors = sum(self.responsibility)
        for c in range(self.k):
            # Maximize mixing covariance
            weights[c] = priors[c] / sum(priors)
            # Maximize means
            means[c] = np.sum(
                            np.multiply(
                                self.responsibility[:, c][:, np.newaxis],
                                self.X
                                ),
                            axis=0
                            ) / priors[c]
            # Maximize covariance
            diff = self.X - means[c]
            covs[c] = np.dot(
                        np.multiply(
                            self.responsibility[:, c][:, np.newaxis],
                            diff
                            ).T,
                        diff
                        ) / priors[c]

        return weights, means, covs

    def _log_likelihood(self, x):
        # Calculate Log-likelihood
        probs = np.zeros(self.k)
        for c in range(self.k):
            probs[c] = multivariate_normal.pdf(
                                        x,
                                        mean=self.means[c],
                                        cov=self.covs[c]
                                        ) * self.weights[c]
        prob = np.log(np.sum(probs) + 1e-12)

        return prob

    def predict(self, x:np.ndarray) -> int:
        """
        Args:
            x(np.ndarray): predict data of shape (1, num_features)
        Returns:
            int: predicted class
        Raises:
            ValueError if model has not been converge
        """
        # Check convergence
        if not self.convergence:
            raise ValueError("Fit model first")

        probs = [0 for _ in range(self.k)]
        for c in range(self.k):
            prob = multivariate_normal.pdf(
                                    x,
                                    mean=self.means[c],
                                    cov=self.covs[c]
                                    )
            probs[c] = prob * self.weights[c]
        
        return np.argmax(probs)
