import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression


def fit_linear_regression(X: np.ndarray(), y: np.ndarray()):
    
    reg = LinearRegression().fit(X, y)
    r2score = reg.score(X, y)
    
    return reg, r2score


def iterate_feature_set(input_data: pd.DataFrame(), label_data: pd.DataFrame(), feature_num: int):
    
    log_iteration = []
    
    for label in label_data.columns:
        print(label)
        
        for feature_set in itertools.combinations(input_data.columns, feature_num):
            print(feature_set)
            input_data_selected = input_data[feature_set]
            
            reg, r2score = fit_linear_regression(input_data_selected.values, label_data[label].values)
            log_iteration.append(feature_set + [label, r2score])
    
    return pd.DataFrame(log_iteration, columns=([f'feature_{i+1}' for i in range(feature_num)] + ['label', 'R2score']))