n1 = 0
n2 = 0
n3 = 0
n4 = 0
newLines = ""
with open("weibo_train.txt", "r", encoding="gbk",errors='ignore') as f:
    lines = f.readlines()
    f.close()
    for line in lines:
        label, sentence = line.strip().split("\t")
        if int(label) == 0:
            if n1 < 10000:
                n1 += 1
                newLines += label + "\t" + sentence + "\n"
        if int(label) == 1:
            if n2 < 10000:
                n2 += 1
                newLines += label + "\t" + sentence + "\n"
        if int(label) == 2:
            if n3 < 10000:
                n3 += 1
                newLines += label + "\t" + sentence + "\n"
        if int(label) == 3:
            if n4 < 10000:
                n4 += 1
                newLines += label + "\t" + sentence + "\n"
    with open("small_train.txt", "w") as f2:
        f2.write(newLines)
        f2.close()

