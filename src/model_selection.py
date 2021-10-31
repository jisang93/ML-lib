import numpy as np

from tqdm import tqdm
from typing import Dict

from src.common import performance_eval


# Model selection
class ModelSelection:
    """
    Model Selection with cross validation
    """
    def __init__(self, model):
        """
        Args:
            Model(class): machine learning model
        """
        self.model = model

    def cross_validation(
            self, x: np.ndarray, y: np.ndarray, params_set: Dict, k: int,
            ):
        """
        Args:
            x(np.ndarray): x of train data
            y(np.ndarray): y of train data
            parmas_set(dict): parameter setting for machine learning model
            k(int): number of validation
        Return:
            params(dict): optimal model parameter
        """
        # Set range of parameters
        alpha_list = np.arange(
                        params_set["alpha"][0],
                        params_set["alpha"][1],
                        params_set["step"]
                        )
        beta_list = np.arange(
                        params_set["beta"][0],
                        params_set["beta"][1],
                        params_set["step"]
                        )
        # Split data
        x_split = np.array_split(x, k)
        y_split = np.array_split(y, k)

        best_score = 0
        for alpha in tqdm(alpha_list):
            for beta in beta_list:
                scores = []
                # Check score
                for i in range(k):
                    # Shallow copy of tarin data
                    x_copy = x_split.copy()
                    y_copy = y_split.copy()
                    # Get model with initilization
                    self.model.reset()
                    self.model.alpha = alpha
                    self.model.beta = beta
                    # Pop for test data
                    test_x = x_copy.pop(i)
                    test_y = y_copy.pop(i)
                    # Concatenate for train data
                    train_x = np.concatenate(x_copy)
                    train_y = np.concatenate(y_copy)
                    # Fit model
                    self.model.fit(train_x, train_y)
                    # Prediction
                    pred = [self.model.predict(x) for x in test_x]
                    # Check scores
                    scores.append(performance_eval(test_y, pred, "F1 score"))
                # Evaluate cross validation result
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    params = {
                        "alpha": alpha,
                        "beta": beta,
                        "class_list": self.model.class_list,
                        "class_prior": self.model.class_prior,
                        "class_means": self.model.class_means,
                        "cov": self.model.rda_cov,
                        "learn": True
                        }

        print("Optimal hyper-parameter")
        print(f"alpha : {params['alpha']:.4f}, beta : {params['beta']:.4f}, "
              f"F1 score : {best_score:.4f}")
        print("#" * 50)

        return params
