from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def semdist(ref, hyp, model):
    model = memory
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score)*100 # lower is better

if __name__ == '__main__':
    semdist_model = SentenceTransformer('dangvantuan/sentence-camembert-large')

    ref = "Je suis un homme"
    hyp = "Je suis une femme"
    print(semdist(ref, hyp, semdist_model))