import os


# each metric compute scores for each pair of hypotheses and reference from HATS
# then it compute a correlation between all these metrics



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


def save_scores(metric, name, hats):
    scores = dict()
    for d in hats:
        ref = d["reference"]
        hypA = d["hypA"]
        hypB = d["hypB"]
        scoreA = metric(ref, hypA)
        scoreB = metric(ref, hypB)
        scores["reference", "hypA"] = scoreA
        scores["reference", "hypB"] = scoreB
    with open("results/scores/" + name + ".txt", "w", "utf8") as f:
        for refhyp, hyp in scores.items():
            f.write(refhyp[0] + "\t" + refhyp[1] + "\t" + str(hyp) + "\n")
    return scores

def scores_computed(name):
    return os.path.isfile("results/scores/" + name + ".txt")

def get_scores(metric, name, hats):
    # check if file exists
    if not scores_computed(name):
        print("Computing scores for", name, "...")
        return save_scores(metric, name, hats)
    else:
        print("Reading scores for", name, "...")
        scores = dict()
        with open("results/scores/" + name + ".txt", "r", "utf8") as f:
            for line in f:
                line = line[:-1].split("\t")
                scores[line[0], line[1]] = float(line[2])
        return scores


def semdist(ref, hyp, memory):
    model = memory    
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    score = (1-score) # lower is better
    return score


def wer(ref, hyp, memory):
    return jiwer.wer(ref, hyp)

def cer(ref, hyp, memory):
    return jiwer.cer(ref, hyp)

def phoner(ref, hyp, memory):
    ep = memory
    ref_phon = ep.transliterate(ref)
    hyp_phon = ep.transliterate(hyp)
    return jiwer.cer(ref_phon, hyp_phon)



if __name__ == "__main__":
    hats = read_dataset("hats.txt")
    names = ["wer", "semdist", "cer", "phoner"]
    metric = [wer, semdist, cer, phoner]


    if "phoner" in names and not scores_computed("phoner"):
        import jiwer
        import epitran
        lang_code = 'fra-Latn-p'
        memory = epitran.Epitran(lang_code)
    elif "semdist" in names and not scores_computed("semdist"):
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        # SD_sentence_camembert_large
        model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    elif "wer" in names and not scores_computed("wer") or "cer" in names and not scores_computed("cer"):
        import jiwer

    all_scores = dict()
    for name, metric in zip(names, metric):
        all_scores[name] = get_scores(metric, name, hats)
        