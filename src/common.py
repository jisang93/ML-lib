import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import requests

from io import BytesIO
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score


# Data loading
def load_dataset(path: str, feature_num: int) -> pd.DataFrame:
    """
    Args:
        path: relative path of data file
        feature_num: the number of freature
    Retruns:
        data(dataFrame)
    """
    if ".csv" in path:
        return pd.read_csv(path, header=0)
    else:
        col = ["Class Label"]
        col.extend([f"Feature {i}" for i in range(1, feature_num+1)])

        return pd.read_excel(path, header=0, usecols=col)


# Data eavaluation
def performance_eval(Y, pred, type="accuracy"):
    """
    Args:
        Y(np.ndarray): array of true label
        pred(np.ndarray): array of predictions
        type(str): accuracy / precision / recall / F1 score
    Returns:
        score of performance evaluation
    """
    if type == "accuracy":
        return accuracy_score(Y, pred)
    elif type == "precision":
        return precision_score(Y, pred, average="macro")
    elif type == "recall":
        return recall_score(Y, pred, average="macro")
    elif type == "F1 score":
        return f1_score(Y, pred, average="macro")
    else:
        raise ValueError("Wrong type")


# Data exploration
def check_data(
        df: pd.DataFrame, feature_num: int, graph_type: str = "hist"
        ) -> plt:
    """
    Arags:
        df: dataFrame
        feature_num: number of feature (e.g. 1, 2, 3, ...)
            if you input feature_num -1,
                you can check all data
        graph_type: hist (histogram) / box (boxplot)
    Returns:
        plot of data distribution
    """
    columns = \
        [int(re.findall("[0-9]+", c)[0]) for c in df.columns if "Class" not in c]
    columns.append(-1)
    if feature_num in columns:
        plt.figure(figsize=(8, 6))
        if graph_type == "hist":
            if feature_num != -1:
                # Histogram
                plt.hist(df[f"Feature {feature_num}"])
                plt.ylabel("Frequency")
                plt.xlabel(f"Range of feature {feature_num}")
                plt.title(f"Histogram of feature {feature_num}")
            else:
                raise ValueError("Do not supoort check all data for histogram")
        if graph_type == "box":
            if feature_num == -1:
                # All of boxplot
                plt.boxplot(df.iloc[:, 1:])
                plt.ylabel("Range of feature")
                plt.xlabel(f"Type of feature")
                plt.title(f"Boxplot of feature")
            else:
                # Boxplot
                plt.boxplot(df[f"Feature {feature_num}"])
                plt.ylabel("Range of feature")
                plt.xlabel(f"Type of feature")
                plt.title(f"Boxplot of feature {feature_num}")

        return plt.show()
    else:
        raise ValueError("Feature number is not in data")


# Quantile for data preprocessing
def apply_quantile(x: float, low_q: float, high_q: float) -> float:
    """
    Args:
        x(float): input value
        low_q(float): low quantile value
        high_q(float): high qunatile value
    Returns:
        x(float): output value
    """
    if x < low_q:
        return low_q
    elif x > high_q:
        return high_q
    else:
        return x


# Normalization for data
def apply_normalization(x: float, min_v: float, max_v: float) -> float:
    """
    Args:
        x(float): input value
        min_v(float): min value of column
        max_v(float): max value of column
    Returns:
        x(float): output value
    """
    return (x - min_v) / (max_v - min_v)


# data prerpocessing
def preprocessing(
        data: pd.DataFrame, q_ratio: float, mode: str
        ) -> pd.DataFrame:
    """
    Args:
        data(pd.DataFrame): input data with label(label should be first column)
        q_ratio(int): ratio of quantile
        mode(str): train/test
    Returns:
        data(pd.DataFrame): output of Preprocessing
    """
    # Set index with label
    data = data.set_index(data.columns[0])
    file_path = "./params/feature_info.json"

    # Train mode
    if mode == "train":
        f_info = {}
        for c in data.columns:
            low_q = data[c].quantile(q_ratio)
            high_q = data[c].quantile(1 - q_ratio)
            # Replace outlier value to quantile value
            data.loc[:, c] = \
                data.loc[:, c].apply(lambda x: apply_quantile(x, low_q, high_q))
            # Apply normalization: Log scaling
            data.loc[:, c] = \
                data.loc[:, c].apply(lambda x: apply_normalization(x, low_q, high_q))
            temp = {"low": low_q, "high": high_q}
            f_info[c] = temp
        # Save feature information
        with open(file_path, "w") as f:
            json.dump(f_info, f)
    # Test mode
    elif mode == "test":
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                f_info = json.load(f)
        else:
            url = \
                "https://raw.github.com/jisang93/ML-lib/master/params/feature_info.json"
            file = BytesIO(requests.get(url).content)
            f_info = json.load(file)
        for c in data.columns:
            # Replace outlier value to quantile value
            data.loc[:, c] = \
                data.loc[:, c].apply(
                    lambda x: apply_quantile(x, f_info[str(c)]["low"], f_info[str(c)]["high"])
                    )
            # Apply normalization: Log scaling
            data.loc[:, c] = \
                data.loc[:, c].apply(
                    lambda x: apply_normalization(x, f_info[str(c)]["low"], f_info[str(c)]["high"])
                    )
    # Mode input error
    else:
        raise ValueError("Wrong mode")

    # Apply normalization: Z-score //
    # Didn't apply Z-score normalization in this project
    # data = (data - data.mean()) / data.std()

    # Return to original form
    data = data.reset_index()

    return data


# Data split
def train_test_split(data: np.ndarray, test_ratio: float) -> np.ndarray:
    """
    Args:
        data(np.ndarray): data (class column sholud be 0)
        test_ratio(float): ratio of test data
    Returns:
        X_train(np.ndarray), X_test(np.ndarray),
        Y_train(np.ndarray), Y_test(np.ndarray)
    """
    np.random.shuffle(data)
    test, train = np.vsplit(data, [int(len(data)*test_ratio)])
    y_train, x_train = np.hsplit(train, [1])
    y_test, x_test = np.hsplit(test, [1])

    return x_train, x_test, np.concatenate(y_train), np.concatenate(y_test)
