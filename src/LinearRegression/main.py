import argparse
import os
from LinearRegression import *


def main():
    
    logger = get_logger()
    
    logger.info('read config')
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file (*.yml or *.yaml)')
    args = parser.parse_args()
    
    config_path = args.config
    config = read_config(config_path)

    output_subdir_path = specify_output_path(config)
    os.makedirs(f'{output_subdir_path}/figure', exist_ok=True)
    os.makedirs(f'{output_subdir_path}/model', exist_ok=True)

    logger.info('load dataset')
    input_data, label_data = load_datasets(config)

    logger.info('optimize linear regression models')
    feature_num = config['feature_num']
    if (feature_num != -1):
        optimize_feature_set(input_data, label_data, feature_num, output_subdir_path)
    else:
        use_all_features(input_data, label_data, output_subdir_path)

    logger.info('visualize results')
    if (feature_num == -1):
        feature_num = input_data.shape[1]
    visualize_results(input_data, label_data, feature_num, output_subdir_path, config['figure_format'])