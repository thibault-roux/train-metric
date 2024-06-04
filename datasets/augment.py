


def read_new_data():
    sentences = []
    with open("new_dataset.txt", "r", encoding="utf8") as file:
        for line in file:
            for sentence in line[:-1].split("\t"):
                sentences.append(sentence)
    return sentences

def read_hats(option):
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("hats_annotation_" + option + ".txt", "r", encoding="utf8") as file:
        # next(file) # reference	hypA	hypB	annotation
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["hypB"] = line[2]
            dictionary["annotation"] = float(line[3])
            dataset.append(dictionary)
    return dataset


if __name__ == "__main__":
    pass