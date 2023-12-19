import os
from LinearRegression.preprocess import *
from LinearRegression.LinearRegression import *


config_path = input()
config = read_config(config_path)

output_figure_dir_path, output_model_dir_path = specify_output_path(config)
os.makedirs(output_figure_dir_path, exist_ok=True)
os.makedirs(output_model_dir_path, exist_ok=True)

input_data, label_data = load_datasets(config)

optimize_feature_set(input_data, label_data, config['feature_num'], config['output_dir_path'], output_figure_dir_path, output_model_dir_path)