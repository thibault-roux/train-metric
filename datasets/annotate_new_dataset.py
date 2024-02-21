import jiwer
import epitran
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import progressbar


def read_new_dataset(namefile):
    with open(namefile, "r") as f:
        refs = []
        hypsA = []
        hypsB = []
        err = 0
        for line in f:
            try:
                ref, hypA, hypB = line.strip().split("\t")
            except ValueError:
                err += 1
                continue
            refs.append(ref)
            hypsA.append(hypA)
            hypsB.append(hypB)
    print("skipped:", err)
    return refs, hypsA, hypsB


def semdist(ref, hyp, model):   
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    score = (1-score) # lower is better
    return score


def wer(ref, hyp):
    return jiwer.wer(ref, hyp)

def cer(ref, hyp):
    return jiwer.cer(ref, hyp)

def phoner(ref, hyp, ep):
    ref_phon = ep.transliterate(ref)
    hyp_phon = ep.transliterate(hyp)
    return jiwer.cer(ref_phon, hyp_phon)

def weighted_score(ref, hyp, model, ep, weights):
    # weight scores
    semdist_score = semdist(ref, hyp, model)
    wer_score = wer(ref, hyp)
    cer_score = cer(ref, hyp)
    phoner_score = phoner(ref, hyp, ep)
    score = weights[0]*wer_score + weights[1]*semdist_score + weights[2]*cer_score + weights[3]*phoner_score
    return score



if __name__ == "__main__":
    refs, hypsA, hypsB = read_new_dataset("new_dataset.txt")

    print("Loading models...")

    # phoner
    lang_code = 'fra-Latn-p'
    ep = epitran.Epitran(lang_code)

    # semdist
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')

    print("Computing scores and writing results...")
    # wer, semdist, cer, phoner
    weights = [0.7, 5, 5, 5]
    with open("extended_hats_annotation.txt", "w", encoding="utf8") as f:
        bar = progressbar.ProgressBar(maxval=len(refs))
        for i in range(len(refs)):
            bar.update(i)
            scoreA = weighted_score(refs[i], hypsA[i], model, ep, weights)
            scoreB = weighted_score(refs[i], hypsB[i], model, ep, weights)

            # 1 means that A is the best, 0 means that B is the best
            if scoreA < scoreB: # if A is better
                txt = refs[i] + "\t" + hypsA[i] + "\t" + hypsB[i] + "\t1\n"
            elif scoreA > scoreB:
                txt = refs[i] + "\t" + hypsA[i] + "\t" + hypsB[i] + "\t0\n"
            else:
                txt = refs[i] + "\t" + hypsA[i] + "\t" + hypsB[i] + "\t0.5\n"
            f.write(txt)
    print("Done!")