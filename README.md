# pipeline_LinearRegression
## Installation
```
$ git clone https://github.com/hirotak22/pipeline_LinearRegression.git
$ cd pipeline_LinearRegression
$ pip install .
```
or 
```
$ pip install git+https://github.com/hirotak22/pipeline_LinearRegression
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
figure_settings: 
    format: "pdf"
    show_features: True
    show_score: True
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
#### figure_settings
- 出力する図についての設定
- `format`は図の形式を指定し、デフォルトは`png`
- `show_features`は図のタイトル部分に説明変数の一覧を含めるかどうかを指定し、デフォルトは`True`
- `show_score`は図のタイトル部分に$R^2$およびadjusted$R^2$を含めるかどうかを指定し、デフォルトは`True`