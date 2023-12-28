import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
import pickle
import json


def adjust_r2score(r2score: float, sample_num: int, feature_num: int):
    return 1 - ((1 - r2score) * (sample_num - 1) / (sample_num - feature_num - 1))


def fit_linear_regression(X: np.ndarray, y: np.ndarray):
    
    reg = LinearRegression().fit(X, y)
    r2score = reg.score(X, y)
    
    return reg, r2score


def iterate_feature_set(input_data: pd.DataFrame, label_data: pd.DataFrame, feature_num: int):
    
    log_iteration = []
    
    sample_num = len(input_data)
    
    for label in label_data.columns:
        for feature_set in itertools.combinations(input_data.columns.to_list(), feature_num):
            feature_set = list(feature_set)
            input_data_selected = input_data[feature_set]
            
            reg, r2score = fit_linear_regression(input_data_selected.values, label_data[label].values)
            adjusted_r2score = adjust_r2score(r2score, sample_num, feature_num)
            
            log_iteration.append(feature_set + [label, r2score, adjusted_r2score])
    
    return pd.DataFrame(log_iteration, columns=([f'feature_{i+1}' for i in range(feature_num)] + ['label', 'R2score', 'adjusted_R2score']))


def summarize_iteration_result(input_data: pd.DataFrame, label_data: pd.DataFrame, feature_num: int, output_subdir_path: str):
    
    log_iteration = iterate_feature_set(input_data, label_data, feature_num)
    log_iteration.to_csv(f'{output_subdir_path}/log_iteration.csv', header=True, index=False)
    
    return log_iteration


def optimize_feature_set(input_data: pd.DataFrame, label_data: pd.DataFrame, feature_num: int, output_subdir_path: str):
    
    log_iteration = summarize_iteration_result(input_data, label_data, feature_num, output_subdir_path)
    
    optimization_result = {}
    sample_num = len(input_data)
    
    for label in label_data.columns:
        log_iteration_extracted = log_iteration.query(f'label == "{label}"').copy()
        log_iteration_extracted.sort_values('R2score', ascending=False, inplace=True)
        best_feature_set = log_iteration_extracted.iloc[0, :feature_num].to_list()
        
        best_reg, best_r2score = fit_linear_regression(input_data[best_feature_set].values, label_data[label].values)
        output_model_path = f'{output_subdir_path}/model/best_model_{label}.pkl'
        pickle.dump(best_reg, open(output_model_path, mode='wb'))
        
        best_adjusted_r2score = adjust_r2score(best_r2score, sample_num, feature_num)
        
        optimization_result[label] = {'feature_set': best_feature_set, 'R2score': best_r2score, 'adjusted_R2score': best_adjusted_r2score, 'model_path': output_model_path}
    
    json.dump(optimization_result, open(f'{output_subdir_path}/optimization_result.json', mode='w'), indent=2)
    
    return None


def use_all_features(input_data: pd.DataFrame, label_data: pd.DataFrame, output_subdir_path: str):
    
    result = {}

    for label in label_data.columns:
        reg, r2score = fit_linear_regression(input_data.values, label_data[label].values)
        output_model_path = f'{output_subdir_path}/model/model_{label}.pkl'
        pickle.dump(reg, open(output_model_path, mode='wb'))
        
        result[label] = {'feature_set': input_data.columns.to_list(), 'R2score': r2score, 'model_path': output_model_path}
    
    json.dump(result, open(f'{output_subdir_path}/result.json', mode='w'), indent=2)
    
    return None


def inference(model_path: str, X: np.ndarray):
    
    reg = pickle.load(open(model_path, mode='rb'))
    preds = reg.predict(X)
    
    return preds


def compute_r2score_and_adjusted_r2score(model_path: str, X: np.ndarray, y: np.ndarray):
    
    reg = pickle.load(open(model_path, mode='rb'))
    r2score = reg.score(X, y)
    sample_num, feature_num = X.shape
    adjusted_r2score = adjust_r2score(r2score, sample_num, feature_num)
    
    return r2score, adjusted_r2score