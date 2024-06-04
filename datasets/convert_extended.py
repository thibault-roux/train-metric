# does the opposite of convert.py: takes the extended_hats_annotation.txt file and convert it to the original format but with 1 or 0 since no humans annotate this dataset.

def read_hats():
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("extended_hats_annotation.txt", "r", encoding="utf8") as file:
        # next(file) # reference	hypA	hypB	annotation
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["hypB"] = line[2]
            dictionary["annotation"] = float(line[3])
            if dictionary["annotation"] == 0:
                dictionary["nbrA"] = 0
                dictionary["nbrB"] = 1
            elif dictionary["annotation"] == 1:
                dictionary["nbrA"] = 1
                dictionary["nbrB"] = 0
            else:
                dictionary["nbrA"] = 0.5
                dictionary["nbrB"] = 0.5
            dataset.append(dictionary)
    return dataset

def write_new_hats(dataset):
    with open("hats_extended.txt", "w", encoding="utf8") as file:
        # reference	hypA	nbrA	hypB	nbrB
        file.write("reference\thypA\tnbrA\thypB\tnbrB\n")
        for item in dataset:
            file.write(item["ref"] + "\t" + item["hypA"] + "\t" + str(item["nbrA"]) + "\t" + item["hypB"] + "\t" + str(item["nbrB"]) + "\n")

if __name__ == "__main__":
    option = "_train_best" # ""
    dataset = read_hats()
    write_new_hats(dataset)
    print("done")