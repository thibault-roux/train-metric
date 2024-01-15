



def read_dataset(dataname):
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open(dataname, "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["reference"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["nbrA"] = int(line[2])
            dictionary["hypB"] = line[3]
            dictionary["nbrB"] = int(line[4])
            dataset.append(dictionary)
    return dataset

def read_hats_annotation(subset): # subset = {train, test}
    dataset = []
    with open("hats_annotation_" + subset + ".txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["reference"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["hypB"] = line[2]
            dataset.append(dictionary)
    return dataset

def convert_to_normal_format(subset): # subset = {train, test}
    subset_dataset = read_hats_annotation(subset)
    full_dataset = read_dataset("hats.txt")
    normal_dataset = []
    for subset_item in subset_dataset:
        for full_item in full_dataset:
            if subset_item["reference"] == full_item["reference"] and subset_item["hypA"] == full_item["hypA"] and subset_item["hypB"] == full_item["hypB"]:
                normal_dataset.append(full_item)
                break
    # write normal dataset
    with open("hats_" + subset + ".txt", "w", encoding="utf8") as file:
        file.write("reference\thypA\tnbrA\thypB\tnbrB\n")
        for item in normal_dataset:
            file.write(item["reference"] + "\t" + item["hypA"] + "\t" + str(item["nbrA"]) + "\t" + item["hypB"] + "\t" + str(item["nbrB"]) + "\n")
    

if __name__ == "__main__":
    convert_to_normal_format("train")
    convert_to_normal_format("test")