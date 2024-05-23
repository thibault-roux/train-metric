


def read_hats(namefile):
    dataset = []
    with open(namefile + ".txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["reference"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["nbrA"] = line[2]
            dictionary["hypB"] = line[3]
            dictionary["nbrB"] = line[4]
            dataset.append(dictionary)
    return dataset

def write(namefile, dataset):
    with open(namefile + ".txt", "w", encoding="utf8") as file:
        file.write("reference\thypA\tnbrA\thypB\tnbrB\n")
        for item in dataset:
            file.write(f"{item['reference']}\t{item['hypA']}\t{item['nbrA']}\t{item['hypB']}\t{item['nbrB']}\n")


if __name__ == "__main__":
    namefile = "hats_train"

    dataset = read_hats(namefile)

    # filter dataset with 0
    dataset_easy = []
    dataset_tough = []
    dataset_very_tough = []
    dataset_best = []
    for item in dataset:
        A = int(item["nbrA"])
        B = int(item["nbrB"])
        agreement = max(A, B) / (A + B)
        if A == "0" or B == "0":
            dataset_easy.append(item)
        else:
            dataset_tough.append(item)
            
            if agreement < 0.75:
                dataset_very_tough.append(item)
        if agreement > 0.7:
            dataset_best.append(item)

    # write(namefile + "_easy", dataset_easy)
    # write(namefile + "_tough", dataset_tough)
    # write(namefile + "_very_tough", dataset_very_tough)
    write(namefile + "_best", dataset_best)