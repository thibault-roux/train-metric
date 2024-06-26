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

def inference_semdist(text, memory):
    tokenizer, model = memory
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def semdist2(ref, hyp, memory):
    ref_projection = inference_semdist(ref, memory).reshape(1, -1)
    hyp_projection = inference_semdist(hyp, memory).reshape(1, -1)
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

def inference_test(dataset, namefile, epoch, model_name, train_data, certitude):
    if model_name == "multi":
        pretrained_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    elif model_name == "french":
        pretrained_model_name = 'dangvantuan/sentence-camembert-large'
    else:
        print("Model name not recognized:", model_name)
        exit(-1)

    if train_data != "none":
        print("Testing the fine-tuned model...")
        # Set up the Siamese network and the dataset
        max_length = 30
        siamese_network = SiameseNetwork(pretrained_model_name, max_length)
        # Load the last saved pretrained model if available
        saved_model_path = 'models/' + model_name + "/" + train_data + '/model.pth.' + str(epoch)
        if os.path.exists(saved_model_path):
            siamese_network.load_state_dict(torch.load(saved_model_path))
            print(f"Loaded pretrained model from {saved_model_path}")
        else:
            print(f"No pretrained model found at {saved_model_path}")
            exit(-1)
        model = siamese_network.camembert
    elif train_data == "none":
        print("Testing the original model...")
        model = AutoModel.from_pretrained(pretrained_model_name)
    else:
        print("Train data not recognized:", train_data)
        exit(-1)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    memory = (tokenizer, model)

    print("Evaluating...")
    x_score = evaluator(semdist2, dataset, memory=memory, certitude=certitude)
    
    print()
    print(x_score)

    with open("results/final_correlation.txt", "a", encoding="utf8") as file:
        clean_pretrained_model_name = pretrained_model_name.split("/")[1]
        txt = namefile + "," + str(epoch) + "," + clean_pretrained_model_name + "," + train_data + "," + str(certitude*100) + "%," + str(x_score) + "\n"
        file.write(txt)


def specific_epoch(epoch):
    # dataset = read_dataset("hats_test.txt")
    dataset = read_dataset("hats.txt")
    inference_test(dataset, epoch, test_new_model=True, certitude=1)



if __name__ == '__main__':
    namefiles = ["hats_test.txt"] # , "hats.txt"] # for dataset
    model_names = ["multi"] # french or multi
    train_datas = ["hats_train_best"] # "none" or "hats_extended" or "hats_train" or "hats_train_best"
    certitudes = [1] # 0 or 1 or 0.7
    epoch_begin = 1
    epochs = 39

    for namefile in namefiles:
        dataset = read_dataset(namefile)
        for model_name in model_names:
            for train_data in train_datas:
                for certitude in certitudes:
                    if train_data != "none":
                        for epoch in range(epoch_begin, epochs):
                            print("\n\n-----------------")
                            print("namefile:", namefile)
                            print("epoch:", epoch)
                            print("model_name:", model_name)
                            print("train_data:", train_data)
                            print("certitude:", certitude)
                            inference_test(dataset=dataset, namefile=namefile, epoch=epoch, model_name=model_name, train_data=train_data, certitude=certitude)
                    else:
                        print("\n\n-----------------")
                        print("namefile:", namefile)
                        print("epoch:", epochs-1)
                        print("model_name:", model_name)
                        print("train_data:", train_data)
                        print("certitude:", certitude)
                        inference_test(dataset=dataset, namefile=namefile, epoch=epochs-1, model_name=model_name, train_data=train_data, certitude=certitude)