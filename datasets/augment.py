# import random to take a random element from a list
import random


def read_data_for_augmentation():
    sentences = []
    with open("new_dataset.txt", "r", encoding="utf8") as file:
        for line in file:
            for sentence in line[:-1].split("\t"):
                sentences.append(sentence)
    return sentences

def read_hats(option):
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("hats" + option + ".txt", "r", encoding="utf8") as file:
        # next(file) # reference	hypA	nbrA    hypB    nbrB
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["nbrA"] = float(line[2])
            dictionary["hypB"] = line[3]
            dictionary["nbrB"] = float(line[4])
            dataset.append(dictionary)
    return dataset


if __name__ == "__main__":
    option = "_train_best" # ""
    AUGMENTATION = 10
    dataset = read_hats(option)
    data_for_augmentation = read_data_for_augmentation()

    new_dataset = []
    for dictionary in dataset:
        new_dataset.append(dictionary)
        # take a random value from data_for_augmentation
        for i in range(AUGMENTATION):
        random_sentence = random.choice(data_for_augmentation)
        new_dataset.append({"ref": dictionary["ref"], "hypA": dictionary["hypA"], "nbrA": 7, "hypB": random_sentence, "nbrB": 0})
        new_dataset.append({"ref": dictionary["ref"], "hypA": random_sentence, "nbrA": 0, "hypB": dictionary["hypB"], "nbrB": 7})
    with open("hats" + option + "_augmented.txt", "w", encoding="utf8") as file:
        # reference	hypA	nbrA	hypB	nbrB
        file.write("reference\thypA\tnbrA\thypB\tnbrB\n")
        for item in new_dataset:
            file.write(item["ref"] + "\t" + item["hypA"] + "\t" + str(item["nbrA"]) + "\t" + item["hypB"] + "\t" + str(item["nbrB"]) + "\n")
    print("done")