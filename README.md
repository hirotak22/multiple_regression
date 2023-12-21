# pipeline_LinearRegression
## Installation
```
$ linearreg ${config}
```

## Usage
```
$ linearreg ${config_file_path}
```

## Config format
### config example
```
input_data_path: "data/input_sample.csv"
label_data_path: "data/label_sample.csv"
output_dir_path: "result"
feature_num: 3
figure_format: "png"
```
#### input_data_path
- 入力データのパス
#### label_data_path
- ラベルデータのパス
#### output_dir_path
- 出力先のディレクトリのパス
#### feature_num
- 線形回帰に用いる説明変数の数
- `-1`に設定するとすべての特徴量を説明変数として使用する
- デフォルトでは`-1`
#### figure_format
- 出力する図の拡張子
- デフォルトでは`png`