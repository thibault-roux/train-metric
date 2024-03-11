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


def save_scores(metric, name, hats, memory):
    scores = dict()
    for d in hats:
        ref = d["reference"]
        hypA = d["hypA"]
        hypB = d["hypB"]
        scoreA = metric(ref, hypA, memory)
        scoreB = metric(ref, hypB, memory)
        scores[d["reference"], d["hypA"]] = scoreA
        scores[d["reference"], d["hypB"]] = scoreB
    with open("results/scores/" + name + ".txt", "w", encoding="utf8") as f:
        for refhyp, score in scores.items():
            f.write(refhyp[0] + "\t" + refhyp[1] + "\t" + str(score) + "\n")
    return scores

def scores_computed(name):
    return os.path.isfile("results/scores/" + name + ".txt")

def get_scores(metric, name, hats, memory):
    # check if file exists
    if not scores_computed(name):
        print("Computing scores for", name, "...")
        return save_scores(metric, name, hats, memory)
    else:
        print("Reading scores for", name, "...")
        scores = dict()
        with open("results/scores/" + name + ".txt", "r", encoding="utf8") as f:
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

    # choice of metrics
    names = ["wer", "semdist", "cer", "phoner"]
    metrics = [wer, semdist, cer, phoner]
    memories = [0] * len(names)


    if "phoner" in names and not scores_computed("phoner"):
        import jiwer
        import epitran
        lang_code = 'fra-Latn-p'
        memory = epitran.Epitran(lang_code)
        memories[names.index("phoner")] = memory
    if "semdist" in names and not scores_computed("semdist"):
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        # SD_sentence_camembert_large
        memory = SentenceTransformer('dangvantuan/sentence-camembert-large')
        memories[names.index("semdist")] = memory
    if "wer" in names and not scores_computed("wer") or "cer" in names and not scores_computed("cer"):
        import jiwer

    # getting scores
    all_scores = dict()
    for name, metric, memory in zip(names, metrics, memories):
        all_scores[name] = get_scores(metric, name, hats, memory)

    # change format to a dictionary of list
    all_scores_lists = {name: list(all_scores[name].values()) for name in names}

    # compute inter-correlations
    from scipy import stats

    txt = ","
    for name in names:
        txt += name + ","
    txt = txt[:-1]
    for name1 in names:
        txt = txt[:-1] + "\n" + name1 + ","
        for name2 in names:
            corr = stats.spearmanr(all_scores_lists[name1], all_scores_lists[name2])
            print(name1, name2, corr[0])
            txt += str(corr[0]) + ","
    with open("results/correlations.txt", "w", encoding="utf8") as f:
        f.write(txt)