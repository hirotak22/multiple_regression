import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from LinearRegression.LinearRegression import inference, compute_r2score_and_adjusted_r2score


def visualize_result(ys: np.ndarray, preds: np.ndarray, label: str, feature_set: list, r2score: float, adjusted_r2score: float, output_subdir_path: str, format: str, show_features: bool, show_score: bool):
    
    features = ','.join(feature_set)
    
    title = label
    if show_features:
        title += f'\nfeatures: {features}'
    if show_score:
        title += f'\n$R^2$ = {r2score:.5e}\tadjusted $R^2$ = {adjusted_r2score:.5e}'
    
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
    ax.scatter(ys, preds)
    ax.set(xlabel='true', ylabel='preds', title=title)
    fig.savefig(f'{output_subdir_path}/figure/scatterplot_{label}.{format}', bbox_inches='tight')
    plt.close()
    
    return None


def visualize_results(input_data: pd.DataFrame, label_data: pd.DataFrame, feature_num: int, output_subdir_path: str, format: str, show_features: bool, show_score: bool):
    
    if feature_num != -1:
        result = json.load(open(f'{output_subdir_path}/optimization_result.json'))
    else:
        result = json.load(open(f'{output_subdir_path}/result.json'))
    
    for label in result.keys():
        ys = label_data[label].values
        feature_set = result[label]['feature_set']
        X = input_data[feature_set].values
        preds = inference(result[label]['model_path'], X)
        
        r2score, adjusted_r2score = compute_r2score_and_adjusted_r2score(result[label]['model_path'], X, ys)
        
        visualize_result(ys, preds, label, feature_set, r2score, adjusted_r2score, output_subdir_path, format, show_features, show_score)
    
    return None