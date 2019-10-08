# 基于LSTM的中文多分类情感分析

> 基于keras深度学习库，搭建LSTM网络，来对新浪微博的内容进行情感分析，分成四类情感。

## 数据集
 - 下载地址： [百度网盘](https://pan.baidu.com/s/16c93E5x373nsGozyWevITg#list/path=%2F)
 - 数据概览： 36 万多条，带情感标注 新浪微博，包含 4 种情感，其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条
 - 数据来源： 新浪微博
 - 原数据集： [微博情感分析数据集](https://download.csdn.net/download/turkan/9181661)，网上搜集，具体作者、来源不详
 - 加工处理： 写了个脚本sampling，可以将数据集拆分更小一点，加快实验。

## 项目结构

```
|——data
|    |——sampling.py 数据集拆分脚本
|    |——small_train.txt 拆分后的小数据集
|    |——weibo_train.txt 原数据集
|    |——stopWords.txt 中文停用词
|——model
|    |——my_model.h5 keras训练后保存的模型
|    |——work_dict.pickle 经过预处理分词生成的词频字典
|——predict.py 使用训练好的模型进行预测
|——train.py 基于keras搭建LSTM网络进行训练
|——pretreatment.py 对数据集进行预处理以及分词，并生成字典保存
|——requirements.txt 项目所需依赖

```