import yaml
import os


def is_table(file_path: str):
    
    ext = os.path.splitext(file_path)[1][1:]
    
    return (ext == 'csv' or ext == 'tsv')


def read_config(config_path: str):

    with open(config_path) as yml:
        config = yaml.safe_load(yml)
    
    if ('input_data_path' not in config.keys()):
        raise KeyError('\"input data_path\" is missing')
    else:
        if (not is_table(config['input_data_path'])):
            raise ValueError('input data must be csv or tsv file')
    
    if ('label_data_path' not in config.keys()):
        raise KeyError('\"label data_path\" is missing')
    else:
        if (not is_table(config['label_data_path'])):
            raise ValueError('label data must be csv or tsv file')
    
    if ('output_dir_path' not in config.keys()):
        raise KeyError('\"output_dir_path\" is missing')
    
    if ('feature_num' in config.keys()):
        if (type(config['feature_num']) is not int):
            raise ValueError('\"feature_num\" must be integer')
    else:
        config['feature_num'] = -1

    return config


def makedir(config: dict):
    
    output_dir_path = config['output_dir_path']
    feature_num = config['feature_num']
    
    if (feature_num == -1):
        subdir = 'all'
    else:
        subdir = f'f_{feature_num}'
    
    os.makedirs(f'{output_dir_path}/{subdir}/figure', exist_ok=True)
    os.makedirs(f'{output_dir_path}/{subdir}/model', exist_ok=True)
    os.makedirs(f'{output_dir_path}/{subdir}/output', exist_ok=True)
    
    return None