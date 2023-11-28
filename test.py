import progressbar
import numpy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


def semdist(ref, hyp, memory):
    model = memory
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score)*100 # lower is better


def evaluator(metric, dataset, memory=0, certitude=0.7, verbose=True):
    print("certitude: ", certitude*100)
    ignored = 0
    accepted = 0
    correct = 0
    incorrect = 0
    egal = 0

    if verbose:
        bar = progressbar.ProgressBar(max_value=len(dataset))
    for i in range(len(dataset)):
        # if i > 100:
        #     break
        if verbose:
            bar.update(i)
        nbrA = dataset[i]["nbrA"]
        nbrB = dataset[i]["nbrB"]
        
        if nbrA+nbrB < 5:
            ignored += 1
            continue
        maximum = max(nbrA, nbrB)
        c = maximum/(nbrA+nbrB)
        if c >= certitude: # if humans are certain about choice
            accepted += 1
            scoreA = metric(dataset[i]["reference"], dataset[i]["hypA"], memory=memory)
            scoreB = metric(dataset[i]["reference"], dataset[i]["hypB"], memory=memory)
            if (scoreA < scoreB and nbrA > nbrB) or (scoreB < scoreA and nbrB > nbrA):
                correct += 1
            elif scoreA == scoreB:
                egal += 1
            else:
                incorrect += 1
            continue
        else:
            ignored += 1
    print()
    return correct/(correct+incorrect+egal)*100


def write(namefile, x, y):
    with open("results/" + namefile + ".txt", "w", encoding="utf8") as file:
        file.write(namefile + "," + str(x) + "," + str(y) + "\n")


if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    cert_X = 1

    # useful for the metric but we do not need to recompute every time
    print("Importing...")


    check_weight = True 
    
    if check_weight:
        model1 = SentenceTransformer('dangvantuan/sentence-camembert-large')
        model1.load_state_dict(torch.load('models/fine_tuned_sentence_transformer.pth'))
        model2 = SentenceTransformer('dangvantuan/sentence-camembert-large')
        # Check if models have the same weights
        are_models_equal = all(p1.equal(p2) for p1, p2 in zip(model1.parameters(), model2.parameters()))
        if are_models_equal:
            print("Model weights are the same.")
        else:
            print("Model weights are different.")
        exit(0)


    # SD_sentence_camembert_large
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    model.load_state_dict(torch.load('models/fine_tuned_sentence_transformer.pth'))
    # model = model.eval()
    memory=model

    print("Evaluating...")
    x_score = evaluator(semdist, dataset, memory=memory, certitude=cert_X)
    
    print(x_score)