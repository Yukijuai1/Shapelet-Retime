<h1><div align = "center"><font size="6"><b>天津大学《神经网络与深度学习》课程设计</b></font></div></h1>
<div align = "center"><font size="6"><b>基于Shapelet的时间序列分类</b></font></div>


## 实验环境

​	计算资源配置：

- CPU：20 Intel(R) Core(TM) i9-10900F CPU @ 2.80GHz
- RAM：128G
- GPU：NVIDIA Corporation TU102 [TITAN RTX]

​	代码运行环境：

- Ubuntu 20.04
- Python 3.7

## 数据集下载

|             数据集             | 训练集大小 | 测试集大小 | 类别 | 长度 |
| :----------------------------: | :--------: | :--------: | :--: | :--: |
|        SyntheticControl        |    300     |    300     |  6   |  60  |
|           Lightning2           |     60     |     61     |  2   | 637  |
|           PowerCons            |    180     |    180     |  2   | 144  |
|    MixedShapesRegularTrain     |    500     |    2425    |  5   | 1024 |
|              Yoga              |    300     |    3000    |  2   | 426  |
|              Car               |     60     |     60     |  4   | 577  |
| ProximalPhalanxOutlineAgeGroup |    400     |    205     |  3   |  80  |
|             Plane              |    105     |    105     |  7   | 144  |
|           ChinaTown            |     20     |    343     |  2   |  24  |
|              Crop              |    7200    |   16800    |  24  |  46  |

1. 下载数据集:https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
2. 解压至项目目录下，命名为`UCRArchive_2018`


## 运行方式

1. 安装依赖包：`pip install -r requirement.txt`
2. 根据`code.sh`训练和测试相应分类器

## 实验结果

|         数据集\正确率          |  KNN   | LearningShapelet |  PPSN  | ShapeNet | Retime |
| :----------------------------: | :----: | :--------------: | :----: | :------: | :----: |
|        SyntheticControl        | 94.33% |      98.33%      | 91.67% |   64%    | 94.33% |
|           Lightning2           | 75.41% |      68.85%      | 72.13% |  65.57%  | 68.85% |
|           PowerCons            | 98.89% |      78.33%      | 92.78% |   85%    | 97.22% |
|    MixedShapesRegularTrain     | 91.22% |                  | 87.92% |  76.08%  | 83.09% |
|              Yoga              | 81.5%  |      64.06%      | 67.97% |  53.56%  | 71.87% |
|              Car               | 78.33% |      68.33%      | 78.33% |  68.33%  |  60%   |
| ProximalPhalanxOutlineAgeGroup | 84.39% |      83.90%      | 84.88% |  82.44%  | 82.43% |
|             Plane              | 97.14% |      98.09%      | 99.05% |  93.33%  | 94.28% |
|           Chinatown            | 45.18% |      82.51%      | 77.55% |   93%    | 93.87% |
|              Crop              | 63.70% |      54.06%      |        |          | 68.67% |
