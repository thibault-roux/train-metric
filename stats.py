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

def accordance_human_perception(hats, metricnames):
    # compute the number of times the human perception is in accordance with the scores
    accordances = [] # contains a list of booleans for each triplet
    for triplet in hats:
        ref = triplet["reference"]
        hypA = triplet["hypA"]
        hypB = triplet["hypB"]
        nbrA = triplet["nbrA"]
        nbrB = triplet["nbrB"]
        accord = [0] * len(metricnames)
        for i, metricname in enumerate(metricnames):
            if scores[ref, hypA] < scores[ref, hypB]:
                if nbrA < nbrB:
                    accord[i] = 1
            elif scores[ref, hypA] > scores[ref, hypB]:
                if nbrA > nbrB:
                    accord[i] = 1
            else:
                if nbrA == nbrB:
                    accord[i] = 1
        accordances.append(accord)
    return accordances
        

if __name__ == "__main__":
    metricnames = ["semdist", "wer", "cer", "phoner"]
    all_scores = dict()
    for metricname in metricnames:
        all_scores[metricname] = get_scores(metricname)

    hats = read_dataset("hats.txt")

    accordances = accordance_human_perception(hats, metricnames)
    for accord in accordances:
        print(accord)