import pickle

import jieba
import numpy as np
from keras.engine.saving import load_model
from keras.preprocessing import sequence

# 加载分词字典
with open('model/word_dict.pickle', 'rb') as handle:
    word2index = pickle.load(handle)

print("加载模型")
model = load_model('model/my_model.h5')

MAX_SENTENCE_LENGTH = 110
INPUT_SENTENCES = ['哈哈哈开心','真是无语，你们怎么搞的','小姐姐，祝你生日快乐','你他妈的有病']
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in  INPUT_SENTENCES:
    words = jieba.cut(sentence)
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
label2word = {1:'愤怒', 2:'厌恶', 3:'低落', 0:'喜悦'}
for x in model.predict(XX):
    print(x)
    x = x.tolist()
    label = x.index(max(x[0], x[1], x[2], x[3]))
    print(label)
    print('{}'.format(label2word[label]))