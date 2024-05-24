from transformers import CamembertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import CamembertModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import progressbar
from sklearn.metrics import accuracy_score
import os

import test



def read_hats(namefile):
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open(namefile, "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["hypB"] = line[2]
            annotation = float(line[3])
            if annotation == 0.0: # nbrA < nbrB | i.e hypB is the best
                dictionary["annotation"] = -1.0
            elif annotation == 1.0: # nbrA > nbrB | i.e hypA is the best
                dictionary["annotation"] = 1.0
            elif annotation == 0.5:
                # dictionary["annotation"] = [0.5]
                continue
            else:
                raise Exception("annotation is not 0.0, 0.5 or 1.0")
            dataset.append(dictionary)
    return dataset


class HATSDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.refs = []
        self.hypsA = []
        self.hypsB = []
        self.labels = []
        for item in dataset:
            self.refs.append(item['ref'])
            self.hypsA.append(item['hypA'])
            self.hypsB.append(item['hypB'])
            self.labels.append(item['annotation'])

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        encoding1 = self.tokenizer(
            self.refs[idx],
            padding='max_length',  # Pad to the maximum length in the batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding2 = self.tokenizer(
            self.hypsA[idx],
            padding='max_length',  # Pad to the maximum length in the batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding3 = self.tokenizer(
            self.hypsB[idx],
            padding='max_length',  # Pad to the maximum length in the batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return ({
            "input_ids1": encoding1["input_ids"].flatten(),
            "attention_mask1": encoding1["attention_mask"].flatten(),
            "input_ids2": encoding2["input_ids"].flatten(),
            "attention_mask2": encoding2["attention_mask"].flatten(),
            "input_ids3": encoding3["input_ids"].flatten(),
            "attention_mask3": encoding3["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        })


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_model_name, max_length):
        super(SiameseNetwork, self).__init__()

        # CamemBERT model
        self.camembert = CamembertModel.from_pretrained(pretrained_model_name)


    def forward(self, input_ids, attention_mask):
        outputs = self.camembert(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation

class SiameseNetworkWithMarginLoss(nn.Module):
    def __init__(self, siamese_network):
        super(SiameseNetworkWithMarginLoss, self).__init__()
        self.siamese_network = siamese_network
        self.margin_loss = nn.MarginRankingLoss(margin=0) # margin=0.5

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, input_ids3, attention_mask3, label):
        output1 = self.siamese_network(input_ids1, attention_mask1)
        output2 = self.siamese_network(input_ids2, attention_mask2)
        output3 = self.siamese_network(input_ids3, attention_mask3)

        # compute cosine similarity between output1 and output2, and between output1 and output3
        similarity_2 = nn.functional.cosine_similarity(output1, output2) # ref and hypA
        similarity_3 = nn.functional.cosine_similarity(output1, output3) # ref and hypB

        loss = self.margin_loss(similarity_2, similarity_3, label)

        return loss


def evaluate_siamese_network(siamese_network, dataloader):
    siamese_network.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        # progressbar 
        bar = progressbar.ProgressBar(max_value=len(dataloader))
        # for batch in dataloader:
        for i, batch in enumerate(dataloader):
            bar.update(i)
            input_ids1 = batch["input_ids1"]
            attention_mask1 = batch["attention_mask1"]
            input_ids2 = batch["input_ids2"]
            attention_mask2 = batch["attention_mask2"]
            input_ids3 = batch["input_ids3"]
            attention_mask3 = batch["attention_mask3"]
            labels = batch["label"]

            output1 = siamese_network(input_ids1, attention_mask1)
            output2 = siamese_network(input_ids2, attention_mask2)
            output3 = siamese_network(input_ids3, attention_mask3)

            similarity_2 = nn.functional.cosine_similarity(output1, output2)
            similarity_3 = nn.functional.cosine_similarity(output1, output3)
            predictions = (similarity_2 > similarity_3).float()
            # convert predictions 0. to -1
            for i, prediction in enumerate(predictions):
                if prediction == 0.0:
                    predictions[i] = -1.0

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy



def train(model_name, train_data, num_epochs):
    if model_name == 'french':
        pretrained_model_name = 'dangvantuan/sentence-camembert-large'
    elif model_name == 'multi':
        pretrained_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    else:
        raise Exception("model_name must be 'french' or 'multi'")
    
    if train_data == 'hats_train':
        hats_train = read_hats("datasets/hats_annotation_train.txt")
    elif train_data == 'hats_extended':
        hats_train = read_hats("datasets/extended_hats_annotation.txt")
    elif train_data == 'hats_train_best':
        hats_train = read_hats("datasets/hats_annotation_train_best.txt")
    else:
        raise Exception("train_data must be 'hats_train' or 'hats_extended'")
    

    tokenizer = CamembertTokenizer.from_pretrained(pretrained_model_name)

    # set dataset
    max_length = 30

    hats_test = read_hats("datasets/hats_annotation_test.txt")
    hats_dataset_train = HATSDataset(hats_train, tokenizer, max_length)
    hats_dataset_test = HATSDataset(hats_test, tokenizer, max_length)

    # Set up the Siamese network and the dataset
    siamese_network = SiameseNetwork(pretrained_model_name, max_length)
    # Load the last saved pretrained model if available
    last_epoch = -1
    for epoch in range(num_epochs):
        saved_model_path = 'models/' + model_name + "/" + train_data + '/model.pth'
        if os.path.exists(saved_model_path + '.' + str(epoch)):
            last_epoch = epoch
    if last_epoch != -1:
        saved_model_path = 'models/' + model_name + "/" + train_data + '/model.pth'
        siamese_network.load_state_dict(torch.load(saved_model_path + '.' + str(last_epoch)))
        print(f"Loaded trained model from local path: {saved_model_path}")
    else:
        print(f"Local trained model not found at {saved_model_path}. Training from pretrained.")
    siamese_with_margin_loss = SiameseNetworkWithMarginLoss(siamese_network)

    # Set up data loader for training
    batch_size = 32
    dataloader = DataLoader(hats_dataset_train, batch_size=batch_size, shuffle=True)
    # Set up data loader for evaluation
    eval_dataloader = DataLoader(hats_dataset_test, batch_size=batch_size, shuffle=False)
    # Set up optimizer
    optimizer = Adam(siamese_with_margin_loss.parameters(), lr=1e-5)

    # Training loop
    losses = []
    best_accuracy = 0
    for epoch in range(last_epoch+1, num_epochs):
        print(epoch)
        bar = progressbar.ProgressBar(max_value=len(dataloader))
        for i, batch in enumerate(dataloader):
            bar.update(i)

            input_ids1 = batch["input_ids1"]
            attention_mask1 = batch["attention_mask1"]
            input_ids2 = batch["input_ids2"]
            attention_mask2 = batch["attention_mask2"]
            input_ids3 = batch["input_ids3"]
            attention_mask3 = batch["attention_mask3"]
            labels = batch["label"]

            optimizer.zero_grad()
            loss = siamese_with_margin_loss(input_ids1, attention_mask1, input_ids2, attention_mask2, input_ids3, attention_mask3, labels)
            loss.backward()
            losses.append(loss.item())
            # print(loss.item())
            optimizer.step()

        # Save the fine-tuned model
        torch.save(siamese_network.state_dict(), saved_model_path + f".{epoch}")





if __name__ == "__main__":
    model_names = ['french'] #, 'multi']
    train_datas = ['hats_extended'] # hats_train', 'hats_extended', 'hats_train_best'

    num_epochs = 40
    for model_name in model_names:
        for train_data in train_datas:
            print("\n\n--------------------")
            print("model_name:", model_name)
            print("train_data:", train_data)
            print("epoch:", num_epochs)
            train(model_name, train_data, num_epochs)