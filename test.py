import progressbar
import numpy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import os


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
            # print()
            # print(scoreA)
            # print(scoreB)
            if (scoreA < scoreB and nbrA > nbrB) or (scoreB < scoreA and nbrB > nbrA):
                correct += 1
                # print("correct")
            elif scoreA == scoreB:
                egal += 1
                # print("equal")
            else:
                incorrect += 1
                # print("incorrect")
            # input()
            continue
        else:
            ignored += 1
    # print()
    return correct/(correct+incorrect+egal)*100


def write(namefile, x, y):
    with open("results/" + namefile + ".txt", "w", encoding="utf8") as file:
        file.write(namefile + "," + str(x) + "," + str(y) + "\n")




# ----------------- Siamese Network -----------------

import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_model_name, max_length):
        super(SiameseNetwork, self).__init__()
        # CamemBERT model
        self.camembert = AutoModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.camembert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation

def inference_test(dataset, epoch, test_new_model, certitude):
    if test_new_model:
        print("Testing the fine-tuned model...")
        # Set up the Siamese network and the dataset
        pretrained_model_name = 'dangvantuan/sentence-camembert-large'
        max_length = 30
        siamese_network = SiameseNetwork(pretrained_model_name, max_length)
        # Load the last saved pretrained model if available
        saved_model_path = 'models/large/fine_tuned_camembert_hats_model.pth.' + str(epoch)
        if os.path.exists(saved_model_path):
            siamese_network.load_state_dict(torch.load(saved_model_path))
            print(f"Loaded pretrained model from {saved_model_path}")
        model = siamese_network.camembert
    else:
        print("Testing the large model...")
        model = AutoModel.from_pretrained('dangvantuan/sentence-camembert-large') # large
    
    tokenizer = AutoTokenizer.from_pretrained('dangvantuan/sentence-camembert-large') # large
    memory = (tokenizer, model)

    print("Evaluating...")
    x_score = evaluator(semdist2, dataset, memory=memory, certitude=certitude)
    
    print()
    print(x_score)

    with open("results/correlation.txt", "a", encoding="utf8") as file:
        file.write(str(epoch) + "\t" + str(x_score) + "\n")


def specific_epoch(epoch):
    dataset = read_dataset("hats_test.txt")
    inference_test(dataset, epoch, test_new_model=True, certitude=1)


if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats_test.txt")

    cert_X = 1

    # useful for the metric but we do not need to recompute every time
    print("Importing...")

    test_new_model = True # test if we use the fine-tuned model or the large one

    for epoch in range(7): # ckpt epoch saved
        print(epoch)
        inference_test(dataset, epoch, test_new_model, cert_X)
        
