import yaml
import os
import pandas as pd


def is_table(file_path: str):
    
    ext = os.path.splitext(file_path)[1][1:]
    
    return (ext == 'csv' or ext == 'tsv')


def read_config(config_path: str):

    with open(config_path) as yml:
        config = yaml.safe_load(yml)
    
    if 'input_data_path' not in config.keys():
        raise KeyError('\'input data_path\' is missing')
    else:
        if not is_table(config['input_data_path']):
            raise ValueError('input data must be csv or tsv file')
    
    if 'label_data_path' not in config.keys():
        raise KeyError('\'label data_path\' is missing')
    else:
        if not is_table(config['label_data_path']):
            raise ValueError('label data must be csv or tsv file')
    
    if 'feature_num' in config.keys():
        if type(config['feature_num']) is not int:
            raise TypeError('\'feature_num\' must be \'int\'')
    else:
        config['feature_num'] = -1
    
    if 'cross_validation' in config.keys():
        cv_settings = config['cross_validation']
        
        if 'savedir' not in cv_settings.keys():
            raise KeyError('\'savedir\' is missing')
        else:
            if type(cv_settings['savedir']) is not str:
                raise TypeError('\'savedir\' must be \'str\'')
        
        if 'n_splits' in cv_settings.keys():
            if type(cv_settings['n_splits']) is not int:
                raise TypeError('\'n_splits\' must be \'int\'')
        else:
            cv_settings['n_splits'] = 5
        
        if 'shuffle' in cv_settings.keys():
            if type(cv_settings['shuffle']) is not bool:
                raise TypeError('\'shuffle\' must be \'bool\'')
        else:
            cv_settings['shuffle'] = True
        
        if 'seed' in cv_settings.keys():
            if type(cv_settings['seed']) is not int:
                raise TypeError('\'seed\' must be \'int\'')
        else:
            cv_settings['seed'] = 42
        
        config['cross_validation'] = cv_settings
    
    else:
        if 'output_dir_path' not in config.keys():
            raise KeyError('\'output_dir_path\' is missing')
        else:
            if type(config['output_dir_path']) is not str:
                raise TypeError('\'output_dir_path\' must be \'str\'')
        
        if 'figure_settings' in config.keys():
            figure_settings = config['figure_settings']
            
            if 'format' in figure_settings.keys():
                if figure_settings['format'] not in ['png', 'ps', 'pdf', 'svg']:
                    raise ValueError('\'format\' must be any of png, ps, pdf, svg')
            else:
                figure_settings['format'] = 'png'
            
            if 'show_features' in figure_settings.keys():
                if type(figure_settings['show_features']) is not bool:
                    raise TypeError('\'show_features\' must be \'bool\'')
            else:
                figure_settings['show_features'] = True
            
            if 'show_score' in figure_settings.keys():
                if type(figure_settings['show_score']) is not bool:
                    raise TypeError('\'show_score\' must be \'bool\'')
            else:
                figure_settings['show_score'] = True
            
            config['figure_settings'] = figure_settings
        
        else:
            config['figure_settings'] = {'format': 'png', 'show_features': True, 'shoe_score': True}

    return config


def specify_output_path(config: dict):
    
    output_dir_path = config['output_dir_path']
    feature_num = config['feature_num']
    
    if feature_num == -1:
        subdir = 'all'
    else:
        subdir = f'f_{feature_num}'
    
    return f'{output_dir_path}/{subdir}'


def load_dataset(dataset_path: str):
    
    ext = os.path.splitext(dataset_path)[1][1:]
    if ext == 'csv':
        dataset = pd.read_csv(dataset_path)
    if ext == 'tsv':
        dataset = pd.read_table(dataset_path)
    
    return dataset


def load_datasets(config: dict):
    
    input_data = load_dataset(config['input_data_path'])
    label_data = load_dataset(config['label_data_path'])
    
    return input_data, label_data