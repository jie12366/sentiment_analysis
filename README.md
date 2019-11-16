# 基于LSTM的中文情绪识别

> 基于keras深度学习库，搭建LSTM网络，来对数据集进行情绪识别，分成六类情绪。

## 数据集
 - 下载地址： https://biendata.com/ccf_tcci2018/datasets/emotion/
 - 数据概览： 4万多条句子,分为其他（Null), 喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness）六类
 - 数据提供方： 清华大学计算机系黄民烈副教授

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
## 情绪识别api
https://github.com/jie12366/sentiment_analysis_api
