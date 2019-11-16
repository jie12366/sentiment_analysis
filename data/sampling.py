import json
newLines = ""
with open("train.json", "r", encoding="utf-8",errors='ignore') as f:
    lines = json.load(f)
    f.close()
    for line in lines:
        sentence = line[0].replace(' ', '')
        label = line[1]
        print(sentence)
        # if int(label) == 0:
        #         newLines += label + "\t" + sentence + "\n"
    # with open("small_train.txt", "w") as f2:
    #     f2.write(newLines)
    #     f2.close()

