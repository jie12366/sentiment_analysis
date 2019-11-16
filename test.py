import jieba

text = jieba.cut('对骂 我 从来 没 怕 过 ， 你们 也 就 只能 考虑 暗杀 了 ， 这样 就 充分 保护 动物 了 ， 臭 傻逼 们 ')  # 进行分词
print(','.join(text))