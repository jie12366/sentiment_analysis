import collections
import pickle
import re
import jieba

# 数据过滤
def regex_filter(s_line):
    # 剔除英文、数字，以及空格
    special_regex = re.compile(r"[a-zA-Z0-9\s]+")
    # 剔除英文标点符号和特殊符号
    en_regex = re.compile(r"[.…{|}#$%&\'()*+,!-_./:~^;<=>?@★●，。]+")
    # 剔除中文标点符号
    zn_regex = re.compile(r"[《》！、，“”；：（）【】]+")

    s_line = special_regex.sub(r"", s_line)
    s_line = en_regex.sub(r"", s_line)
    s_line = zn_regex.sub(r"", s_line)
    return s_line

# 加载停用词
def stopwords_list(file_path):
    stopwords = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
    return stopwords

word_freqs = collections.Counter()  # 词频
stopword = stopwords_list("data/stopWords.txt")
max_len = 0
with open('data/small_train.txt', 'r+', encoding="gbk",errors='ignore') as f:
    lines = f.readlines()
    for line in lines:
        # 取出label和句子
        label, sentence = line.strip("\n").split("\t")
        # 数据预处理
        sentence = regex_filter(sentence)
        words = jieba.cut(sentence)
        x = 0
        for word in words:
            # 去除停用词
            if word not in stopword:
                print(word)
                word_freqs[word] += 1
                x += 1
        max_len = max(max_len, x)
print(max_len)
print('nb_words ', len(word_freqs))

## 准备数据
MAX_FEATURES = 80000 # 最大词频数
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
# 构建词频字典
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
# 将词频字典写入文件中保存
with open('model/word_dict.pickle', 'wb') as handle:
    pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)