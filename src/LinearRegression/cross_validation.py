import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import json
from LinearRegression.LinearRegression import adjust_r2score


def specify_cv_path(config: dict):
    
    cv_savedir_path = config['cross_validation']['savedir']
    feature_num = config['feature_num']
    
    if feature_num == -1:
        subdir = 'all'
    else:
        subdir = f'f_{feature_num}'
    
    return f'{cv_savedir_path}/{subdir}'


def construct_cv_dataset(input_dataset: pd.DataFrame, label_dataset: pd.DataFrame, savedir: str, n_splits: int, shuffle: bool, seed: int):
    
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for i, (train_index, valid_index) in enumerate(kf.split(input_dataset.values)):
        train_input = input_dataset.iloc[train_index]
        train_label = label_dataset.iloc[train_index]
        train_input.to_csv(f'{savedir}/data/train_input_{i+1}.csv', header=True, index=False)
        train_label.to_csv(f'{savedir}/data/train_label_{i+1}.csv', header=True, index=False)
        
        valid_input = input_dataset.iloc[valid_index]
        valid_label = label_dataset.iloc[valid_index]
        valid_input.to_csv(f'{savedir}/data/valid_input_{i+1}.csv', header=True, index=False)
        valid_label.to_csv(f'{savedir}/data/valid_label_{i+1}.csv', header=True, index=False)
    
    return None


def load_cv_datasets(savedir: str, n_cv: int):
    
    train_input = pd.read_csv(f'{savedir}/data/train_input_{n_cv+1}.csv')
    train_label = pd.read_csv(f'{savedir}/data/train_label_{n_cv+1}.csv')
    valid_input = pd.read_csv(f'{savedir}/data/valid_input_{n_cv+1}.csv')
    valid_label = pd.read_csv(f'{savedir}/data/valid_label_{n_cv+1}.csv')
    
    return train_input, train_label, valid_input, valid_label


def fit_linear_regression_for_validation(train_X: np.ndarray, train_y: np.ndarray, valid_X: np.ndarray, valid_y: np.ndarray):
    
    reg = LinearRegression().fit(train_X, train_y)
    train_r2score = reg.score(train_X, train_y)
    valid_r2score = reg.score(valid_X, valid_y)
    
    return reg, train_r2score, valid_r2score


def iterate_feature_set_for_validation(train_input: pd.DataFrame, train_label: pd.DataFrame, valid_input: pd.DataFrame, valid_label: pd.DataFrame, feature_num: int):
    
    log_iteration = []
    log_columns = [f'feature_{i+1}' for i in range(feature_num)] \
        + ['label', 'R2score(train)', 'adjusted_R2score(train)', 'R2score(valid)', 'adjusted_R2score(valid)']
    
    for label in train_label.columns:
        for feature_set in itertools.combinations(train_input.columns.to_list(), feature_num):
            feature_set = list(feature_set)
            train_input_selected = train_input[feature_set]
            valid_input_selected = valid_input[feature_set]
            
            reg, train_r2score, valid_r2score = fit_linear_regression_for_validation(train_input_selected.values, train_label[label].values,
                                                                                     valid_input_selected.values, valid_label[label].values,)
            train_adjusted_r2score = adjust_r2score(train_r2score, len(train_input), feature_num)
            valid_adjusted_r2score = adjust_r2score(valid_r2score, len(valid_input), feature_num)
            
            log_iteration.append(feature_set + [label, train_r2score, train_adjusted_r2score, valid_r2score, valid_adjusted_r2score])
    
    return pd.DataFrame(log_iteration, columns=log_columns)


def summarize_iteration_result_for_cv(cv_savedir_path: str, n_splits: int, feature_num: int):
    
    log_list = []
    for i in range(n_splits):
        train_input, train_label, valid_input, valid_label = load_cv_datasets(f'{cv_savedir_path}/', i)
        log_iteration = iterate_feature_set_for_validation(train_input, train_label, valid_input, valid_label, feature_num)
        log_list.append(log_iteration)
    
    log_iteration_cv = pd.concat(log_list, ignore_index=True)
    log_iteration_cv.to_csv(f'{cv_savedir_path}/result/log_iteration.csv', header=True, index=False)
    
    return log_iteration_cv


def optimize_feature_set_for_cv(cv_savedir_path: str, n_splits: int, feature_num: int):
    
    log_iteration_cv = summarize_iteration_result_for_cv(cv_savedir_path, n_splits, feature_num)
    label_list = log_iteration_cv['label'].unique()
    
    optimization_result = {}
    col_features = [f'feature_{i+1}' for i in range(feature_num)]
    for label in label_list:
        log_iteration_cv_extracted = log_iteration_cv.query(f'label == "{label}"').copy()
        max_valid_r2score = -(2**31)
        for feature_set, df in log_iteration_cv_extracted.groupby(col_features):
            mean_valid_r2score = df['R2score(valid)'].mean()
            if max_valid_r2score < mean_valid_r2score:
                max_valid_r2score = mean_valid_r2score
                best_feature_set = feature_set
        
        log_best_iteration = log_iteration_cv_extracted[(log_iteration_cv_extracted[col_features] == best_feature_set).all(axis=1)].copy()
        train_r2score_list = log_best_iteration['R2score(train)'].to_list()
        train_adjusted_r2score_list = log_best_iteration['adjusted_R2score(train)'].to_list()
        valid_r2score_list = log_best_iteration['R2score(valid)'].to_list()
        valid_adjusted_r2score_list = log_best_iteration['adjusted_R2score(valid)'].to_list()
        
        optimization_result[label] = {'feature_set': best_feature_set,
                                      'R2score(train)': train_r2score_list,
                                      'adjusted_R2score(train)': train_adjusted_r2score_list,
                                      'R2score(valid)': valid_r2score_list,
                                      'adjusted_R2score(valid)': valid_adjusted_r2score_list}
    
    json.dump(optimization_result, open(f'{cv_savedir_path}/result/optimization_result.json', mode='w'), indent=2)
    
    return None
