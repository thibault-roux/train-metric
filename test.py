import progressbar
import numpy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


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


# ----------------- second method -----------------

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


# ----------------- Evaluator -----------------

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
        if i > 300:
            break
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




# ----------------- Siamese Network -----------------

import torch.nn as nn


# Define the Siamese network
class SiameseNetwork(nn.Module):
    def __init__(self, model_name):
        super(SiameseNetwork, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        transformer_output_dim = self.model.config.hidden_size

        self.fc = nn.Sequential(
            nn.Linear(3 * transformer_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # Last layer hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, ref, hyp_a, hyp_b):
        # Tokenize sentences
        encoded_input_ref = self.tokenizer(ref, padding=True, truncation=True, return_tensors='pt')
        encoded_input_hyp_a = self.tokenizer(hyp_a, padding=True, truncation=True, return_tensors='pt')
        encoded_input_hyp_b = self.tokenizer(hyp_b, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output_ref = self.model(**encoded_input_ref)
            model_output_hyp_a = self.model(**encoded_input_hyp_a)
            model_output_hyp_b = self.model(**encoded_input_hyp_b)

        # Perform pooling
        ref_embedding = self.mean_pooling(model_output_ref, encoded_input_ref['attention_mask'])
        hyp_a_embedding = self.mean_pooling(model_output_hyp_a, encoded_input_hyp_a['attention_mask'])
        hyp_b_embedding = self.mean_pooling(model_output_hyp_b, encoded_input_hyp_b['attention_mask'])


        concatenated = torch.cat([ref_embedding, hyp_a_embedding, hyp_b_embedding], dim=1)
        output = self.fc(concatenated)

        return output

    def save_embedding(self, path):
        self.model.save_pretrained(path)


if __name__ == '__main__':
    print("Reading dataset...")
    # dataset = read_dataset("hats.txt")

    cert_X = 1

    # useful for the metric but we do not need to recompute every time
    print("Importing...")


    check_weight = False
    
    if check_weight:

        tokenizer = AutoTokenizer.from_pretrained('dangvantuan/sentence-camembert-large')
        model1 = AutoModel.from_pretrained('./models/hypothesis_classifier')
        memory1 = (tokenizer, model1)

        tokenizer = AutoTokenizer.from_pretrained('dangvantuan/sentence-camembert-large')
        model2 = AutoModel.from_pretrained('dangvantuan/sentence-camembert-large')
        memory2 = (tokenizer, model2)

        model3 = SentenceTransformer('dangvantuan/sentence-camembert-large')
        # model = model.eval()
        # memory=model2

        text = "Voici un premier exemple"
        inf1 = inference_semdist2(text, memory1) # trained
        inf2 = inference_semdist2(text, memory2) # same as model3
        inf3 = model3.encode(text).reshape(1, -1) # base sentence transformer
        print(inf1)
        print(inf2)
        print(inf3)
        
        exit(0)

    tokenizer = AutoTokenizer.from_pretrained('dangvantuan/sentence-camembert-large')
    model = AutoModel.from_pretrained('./models/hypothesis_classifier')
    memory = (tokenizer, model)

    print("Evaluating...")
    x_score = evaluator(semdist2, dataset, memory=memory, certitude=cert_X)
    
    print(x_score)