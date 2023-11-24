

def read_hats():
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("datasets/hats.txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["nbrA"] = int(line[2])
            dictionary["hypB"] = line[3]
            dictionary["nbrB"] = int(line[4])
            if dictionary["nbrA"] > dictionary["nbrB"]:
                dictionary["annotation"] = 1
            elif dictionary["nbrA"] < dictionary["nbrB"]:
                dictionary["annotation"] = 0
            else:
                dictionary["annotation"] = 0.5
            dataset.append(dictionary)
    return dataset

def write_new_hats(dataset):
    with open("datasets/hats_annotation.txt", "w", encoding="utf8") as file:
        file.write("reference\thypA\thypB\tannotation\n")
        for item in dataset:
            file.write(item["ref"] + "\t" + item["hypA"] + "\t" + item["hypB"] + "\t" + str(item["annotation"]) + "\n")

if __name__ == "__main__":
    dataset = read_hats()
    write_new_hats(dataset)
    print("done")