# 简介

这是一个基于pytorch与ReChorus框架的，机器学习论文复现+课程大作业项目

原论文: [Graph Convolution Network based Recommender Systems: Learning Guarantee and Item Mixture Powered Strategy](https://proceedings.neurips.cc/paper_files/paper/2022/hash/18fd48d9cbbf9a20e434c9d3db6973c5-Abstract-Conference.html)

原框架：[ReChorus](https://github.com/THUwangcy/ReChorus)

关键词:

- 图卷积网络(GCN)

- 推荐系统

- 泛化保证

- Item Mixture

- 数据增强

-----------------

# 部署
- **注意**: 本项目的部署对环境有较严格的要求，如有版本相关的报错，可参考`./requirements.txt`中的版本

1. 下载
    ```shell
    git clone https://github.com/GaopwChitu/ML_dazuoye.git <your_project_name>
    ```
    如对`git`的下载不熟悉，也可直接`download`本项目

2. 环境
    ```shell
    pip install -r requirements.txt
    ```
3. 数据集
    数据集需自行下载，然后放于`./data`对应的目录中，其中已有对应的数据处理文件`Amazon.ipynb`与`MovieLens-1M.ipynb`
    本项目使用
    - [Amazon dataset](http://jmcauley.ucsd.edu/data/amazon/links.html) (Grocery_and_Gourmet_Food category, 5-core version with metadata)
    - [MovieLens-1M](https://grouplens.org/datasets/movielens/)

--------------------
# 训练
确保位于本项目根目录，此后
```shell
cd src
```
此后，运行示例
```shell
python main.py --model_name LightGCN --path ../data/ --dataset Grocery_and_Gourmet_Food --gpu 1 
```

- 注意: 本项目在 Windows/Nvidia GPU 上有不明bug，如用n卡加速，易出现`prediction`全`0`的情况



