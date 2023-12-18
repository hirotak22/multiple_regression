import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
import pickle


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


def optimize_feature_set(input_data: pd.DataFrame(), label_data: pd.DataFrame(), feature_num: int, output_dir_path: str):
    
    log_iteration = iterate_feature_set(input_data, label_data, feature_num)
    log_iteration.to_csv(f'{output_dir_path}/log_iteration.csv', header=True, index=False)
    
    for label in label_data.columns:
        log_iteration_extracted = log_iteration.query(f'label == "{label}"')
        log_iteration_extracted.sort_values('R2score', ascending=False, inplace=False)
        best_feature_set = log_iteration_extracted.iloc[1, :feature_num].to_list()
        
        best_reg, best_r2score = fit_linear_regression(input_data[best_feature_set].values, label_data[label].values)
        model_path = f'{output_dir_path}/model/best_model_{label}.pkl'
        pickle.dump(best_reg, model_path)
    
    return None