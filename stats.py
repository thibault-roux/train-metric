import os


def scores_computed(name):
    return os.path.isfile("results/scores/" + name + ".txt")

def get_scores(metricname):
    # check if file exists
    if not scores_computed(metricname):
        raise Exception("Scores for", metricname, "not computed yet.")
    else:
        print("Reading scores for", metricname, "...")
        scores = dict()
        with open("results/scores/" + metricname + ".txt", "r", encoding="utf8") as f:
            for line in f:
                line = line[:-1].split("\t")
                scores[line[0], line[1]] = float(line[2])
        return scores

def read_dataset(dataname):
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("datasets/" + dataname, "r", encoding="utf8") as file:
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

if __name__ == "__main__":
    metricnames = ["semdist", "wer", "cer", "phoner"]
    all_scores = dict()
    for metricname in metricnames:
        all_scores[metricname] = get_scores(metricname)

    hats = read_dataset("hats.txt")