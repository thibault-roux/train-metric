import os
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel

# each metric compute scores for each pair of hypotheses and reference from HATS
# then it compute a correlation between all these metrics



# ----------------- begin fine-tuned -----------------

class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_model_name, max_length):
        super(SiameseNetwork, self).__init__()
        # CamemBERT model
        self.camembert = AutoModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.camembert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation

def load_model(epoch=0):
    pretrained_model_name = 'dangvantuan/sentence-camembert-large'
    max_length = 30
    siamese_network = SiameseNetwork(pretrained_model_name, max_length)
    # Load the last saved pretrained model if available
    saved_model_path = 'models/large/fine_tuned_camembert_hats_model.pth.' + str(epoch)
    if os.path.exists(saved_model_path):
        siamese_network.load_state_dict(torch.load(saved_model_path))
        print(f"Loaded pretrained model from {saved_model_path}")
    else:
        print(f"Model {saved_model_path} not found")
        exit(-1)
    model = siamese_network.camembert
    return model

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def inference_semdist2(text, memory):
    tokenizer, model = memory
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def semdist2(ref, hyp, memory):
    ref_projection = inference_semdist2(ref, memory).reshape(1, -1)
    hyp_projection = inference_semdist2(hyp, memory).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score)*100 # lower is better

# ----------------- end fine-tuned -----------------



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
    names = ["wer", "semdist_trained_0", "cer", "phoner", "semdist"] # "semdist"
    metrics = [wer, semdist2, cer, phoner, semdist] # semdist
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
    if "semdist_trained_0" in names and not scores_computed("semdist_trained_0"):
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        model = load_model(epoch=0)
        tokenizer = AutoTokenizer.from_pretrained('dangvantuan/sentence-camembert-large') # large
        memory = (tokenizer, model)
        memories[names.index("semdist_trained_0")] = memory




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
    for name1 in names:
        txt = txt[:-1] + "\n" + name1 + ","
        for name2 in names:
            corr = stats.spearmanr(all_scores_lists[name1], all_scores_lists[name2])
            print(name1, name2, corr[0])
            txt += str(corr[0]) + ","

    with open("results/inter-correlations.txt", "w", encoding="utf8") as f:
        f.write(txt)