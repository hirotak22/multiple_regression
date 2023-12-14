import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression


def linear_regression(X: np.ndarray(), y: np.ndarray()):
    
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    
    return reg, score


def optimize_feature_set(input_data: pd.DataFrame(), label_data: pd.DataFrame(), feature_num: int):
    
    for feature_set in itertools.combinations(input_data.columns):
        print(feature_set)
    
    return None