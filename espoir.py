from transformers import CamembertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import CamembertModel
from torch.utils.data import DataLoader

tokenizer = CamembertTokenizer.from_pretrained('dangvantuan/sentence-camembert-large')

def read_hats():
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("datasets/hats_annotation.txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["hypB"] = line[2]
            annotation = float(line[3])
            if annotation == 0.0:
                dictionary["annotation"] = [0.0]
            elif annotation == 1.0:
                dictionary["annotation"] = [1.0]
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

# set dataset
max_length = 30
hats = read_hats()
dataset = HATSDataset(hats, tokenizer, max_length)