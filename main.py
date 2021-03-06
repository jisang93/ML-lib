import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests

from io import BytesIO
from sklearn.metrics import confusion_matrix

from src.common import load_dataset, performance_eval
from src.common import preprocessing, train_test_split
from src.model_selection import ModelSelection


# Performance evaluation
class PerformanceEval:
    """
    Evaluation of Model
    """
    def __init__(self, mode, model):
        """
        Args:
            mode(str): train/test
            model: RDA here
        """
        self.mode = mode
        self.model = model
        self.learn = False
        if self.mode == "train":
            self.ms = ModelSelection(model)
        if self.mode == "test":
            self.load_params()

    def train(self, data, params_set, K):
        """
        Args:
            data(str or pd.DataFrame): path of data or dataframe
            params_set(dict): parameter setting for fitting RDA model
            K(int): number of cross validation
        Returns:
            paramters is changed
        """
        # Data load
        if type(data) == "str":
            df = load_dataset(data, 22)
        else:
            df = data
        df = preprocessing(df, 0.15, "train")
        data = np.array(df)
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(data, 0.25)
        # Cross validation for fitting parameters
        params = self.ms.cross_validation(X_train, Y_train, params_set, K)
        self.set_params(params)

        # Calculate prediction score
        pred = []
        for x in X_test:
            pred.append(self.model.predict(x))
        acc_score = performance_eval(Y_test, pred, "accuracy")
        pre_score = performance_eval(Y_test, pred, "precision")
        rec_score = performance_eval(Y_test, pred, "recall")
        F1_score = performance_eval(Y_test, pred, "F1 score")

        print("Prediction score for test data")
        print(f"Accuracy: {acc_score.mean():.4f} / "
              f"Precision: {pre_score.mean():.4f} / "
              f"Recall: {rec_score.mean():.4f} / "
              f"F1 score : {F1_score.mean():.4f}")
        print("#" * 50)

        # Save parameters
        with open("./params/rda-params.pkl", "wb") as f:
            pickle.dump(params, f)
        print("Parameters are saved. File name is 'rda-params.pkl'.")
        print("#" * 50)

        self.learn = True

    def load_params(self):
        """
        Returns:
            Load trained paratmeters setting from github(jisang93/ML-lib)
        """
        url = \
            "https://raw.github.com/jisang93/ML-lib/master/params/rda-params.pkl"
        file = BytesIO(requests.get(url).content)
        params = pickle.load(file)
        self.set_params(params)
        return params

    def set_params(self, params):
        self.model.alpha = params["alpha"]
        self.model.beta = params["beta"]
        self.model.rda_cov = params["cov"]
        self.model.class_means = params["class_means"]
        self.model.class_prior = params["class_prior"]
        self.model.class_list = params["class_list"]
        self.model.learn = params["learn"]
        self.learn = True

    def evaluate(self, data):
        """
        Args:
            data(pd.DataFrame): data for evalution
        Returns:
            matrix(pd.DataFrame): matrix of confusion
        """
        # If select train
        if self.mode == "train" and self.learn is False:
            raise ValueError("You select train mode. Model don't get default"
                             "parameter in train mode.")

        # Data preprocessing
        pre_data = preprocessing(data, 0.15, "test")
        # Set data for eval
        eval_data = np.array(pre_data)
        Y_test, X_test = np.hsplit(eval_data, [1])
        # Calculate prediction score
        pred = []
        for x in X_test:
            pred.append(self.model.predict(x))
        acc_score = performance_eval(Y_test, pred, "accuracy")
        pre_score = performance_eval(Y_test, pred, "precision")
        rec_score = performance_eval(Y_test, pred, "recall")
        F1_score = performance_eval(Y_test, pred, "F1 score")

        # Print prediction score
        print("Prediction score for unseen data")
        print(f"Accuracy: {acc_score.mean():.4f} / "
              f"Precision: {pre_score.mean():.4f} / "
              f"Recall: {rec_score.mean():.4f} / "
              f"F1 score : {F1_score.mean():.4f}")
        return pd.DataFrame(confusion_matrix(Y_test, pred))


# For Checking clusters
def show_results(data, log_likelihood, k, prediction, num_columns):
    """
    Args:
        data(list or np.ndarray): training data(shape=[n_samples, n_features])
        log_likelihood(list): log-likelihood result of fitting
        k(int): the number of clusters
        prediction(np.ndarray): predict reulst
        num_columns(list): index numbers of two columns want to check
    Returns:
        line graph and scatter of data
    """
    # Concatenate source data and prediction
    columns = [f"feature{i+1}" for i in range(data.shape[1])]
    num_columns = [f"feature{i+1}" for i in num_columns]
    df = pd.DataFrame(data, columns=columns)
    df_p = pd.DataFrame(prediction, columns=["class"])
    df = pd.concat([df, df_p], axis=1)
    # Apply graph
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # Log-likelihood
    ax1.plot(log_likelihood)
    ax1.set_title("Log-likelihood")
    ax1.set_xlabel("iterations")
    # Scatter with class
    groups = df.groupby("class")
    for name, group in groups:
        ax2.plot(
            group[num_columns[0]],
            group[num_columns[1]],
            marker="o",
            linestyle="",
            label=name
            )
    ax2.legend()
    ax2.set_title(f"Scatter plot of data (K={k})")
    ax2.set_xlabel("feature 1")
    ax2.set_ylabel("feature 2")

    return plt.show()
