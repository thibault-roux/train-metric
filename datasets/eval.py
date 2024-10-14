

# compare the inference of gpt with the real annotations

def load(namefile):
    with open(namefile, "r") as f:
        next(f)
        data = []
        for line in f:
            line = line.strip().split("\t")
            data.append(line)
    return data

def load_OG():
    with open("./hats.txt", "r") as f:
        next(f)
        agreement = []
        for line in f:
            line = line.strip().split("\t")
            # ref = line[0]
            # hypA = line[1]
            # hypB = line[3]
            nbrA = line[2]
            nbrB = line[4]
            agree = max(nbrA, nbrB)/(nbrA + nbrB)
            agreement.append(agree)
    return agreement

def compare(namefile1, namefile2):
    data1 = load(namefile1)
    data2 = load(namefile2)
    assert len(data1) == len(data2)
    correct = 0
    for i in range(len(data1)):
        if data1[i][3] == data2[i][3]:
            correct += 1
        elif data1[i][3] == "1" and data2[i][3] == "11":
            correct += 1
        elif data1[i][3] == "0" and data2[i][3] == "10":
            correct += 1
    return correct / len(data1)


if __name__ == "__main__":
    namefile1 = "./hats_annotation_split.txt"
    namefile2 = "./hats_annotation_split_gpt.txt"
    print(compare(namefile1, namefile2))
    