# 广告广告
字节跳动广告系统下的穿山甲平台大量招人，有兴趣的直接发简历到我邮箱：xiongyijie.monkjay@bytedance.com。  
也可以直接加我 QQ：2263509062
![ddd](https://user-gold-cdn.xitu.io/2020/6/9/17296d23354015c8?w=1202&h=655&f=png&s=811332)

# 基于LSTM的中文情绪识别

> 基于keras深度学习库，搭建LSTM网络，来对数据集进行情绪识别，分成六类情绪。

## 数据集
 - 下载地址： https://biendata.com/ccf_tcci2018/datasets/emotion/
 - 数据概览： 4万多条句子,分为其他（Null), 喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness）六类
 - 数据来源：数据分别来源于NLPCC Emotion Classification Challenge（训练数据中17113条，测试数据中2242条）和微博数据筛选后人工标注(训练数据中23000条，测试数据中2500条)。
 - 数据提供方： 清华大学计算机系黄民烈副教授

## 项目结构

```
|——data
|    |——train.json 原数据集
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

## 相关博客
[中文情绪识别api](http://jie12366.xyz:8081/#/users/11/articles/46)  

[基于LSTM的中文多分类情感分析](http://jie12366.xyz:8081/#/users/11/articles/35)
