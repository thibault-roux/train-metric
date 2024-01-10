from transformers import CamembertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import CamembertModel
from torch.utils.data import DataLoader
import progressbar


# myenv2

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
                dictionary["annotation"] = [0.0] # [1.0, 0.0]
            elif annotation == 1.0:
                dictionary["annotation"] = [1.0] # [0.0, 1.0]
            elif annotation == 0.5:
                dictionary["annotation"] = [0.5] # [0.0, 0.0] # it is also possible to skip these cases
                # continue
            else:
                raise Exception("annotation is not 0.0, 0.5 or 1.0")
            # dictionary["annotation"] = [float(line[3])]
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
            "label": torch.tensor(self.labels[idx], dtype=torch.float) # should be dtype=torch.long
        })

# set dataset
max_length = 30
hats = read_hats()
dataset = HATSDataset(hats, tokenizer, max_length)



class HypothesisClassifier(nn.Module):
    def __init__(self, camembert_model):
        super(HypothesisClassifier, self).__init__()
        self.camembert = camembert_model
        # self.linear = nn.Linear(in_features=self.camembert.config.hidden_size, out_features=2)
        self.fc = nn.Sequential(
            nn.Linear(3 * self.camembert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1, bias=False), # should be (128, 1)
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, input_ids3, attention_mask3):
        outputs1 = self.camembert(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.camembert(input_ids=input_ids2, attention_mask=attention_mask2)
        outputs3 = self.camembert(input_ids=input_ids3, attention_mask=attention_mask3)
        pooled_output1 = outputs1.pooler_output
        pooled_output2 = outputs2.pooler_output
        pooled_output3 = outputs3.pooler_output
        # logits = self.linear(pooled_output)
        concatenated = torch.cat([pooled_output1, pooled_output2, pooled_output3], dim=1)
        logits = self.fc(concatenated)
        return logits

    def save_embedding(self, path):
        print("Saving in ", path)
        self.camembert.save_pretrained(path)
        print("Saved")

camembert_model = CamembertModel.from_pretrained('dangvantuan/sentence-camembert-base', num_labels=2)
hypothesis_classifier = HypothesisClassifier(camembert_model)

# large = 91.4484, 87.2707
# base = 28.6339, 29.0308
# conclusion: base is three times faster than large


# Define your training parameters
optimizer = torch.optim.Adam(hypothesis_classifier.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()


print("Loading dataloader...")

# DataLoader for training
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # batch_size=32

losses = []

print("dataloader loaded")

import time
# Training loop
num_epochs = 5
eval_scores = []
for epoch in range(num_epochs):
    print(epoch)
    # progress bar
    # bar = progressbar.ProgressBar(max_value=len(dataloader))
    bar = progressbar.ProgressBar(max_value=100)
    # compute the time to process 100 batches
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i> 100:
            print("Time to process 100 batches:", time.time() - start)
            break
        bar.update(i)
        # for batch in dataloader:
        input_ids1 = batch["input_ids1"]
        attention_mask1 = batch["attention_mask1"]
        input_ids2 = batch["input_ids2"]
        attention_mask2 = batch["attention_mask2"]
        input_ids3 = batch["input_ids3"]
        attention_mask3 = batch["attention_mask3"]
        labels = batch["label"]

        optimizer.zero_grad()
        logits = hypothesis_classifier(input_ids1, attention_mask1, input_ids2, attention_mask2, input_ids3, attention_mask3)
        # print("logits:", logits)
        # print("labels:", labels)
        # print("type(logits[0][0]):", type(logits[0][0]))
        # print("type(labels[0][0]):", type(labels[0][0]))
        loss = loss_fn(logits, labels)
        l = loss.item()
        losses.append(l)
        loss.backward()
        optimizer.step() # à remettre

        # for name, param in hypothesis_classifier.named_parameters():
        #     if param.requires_grad:
        #         old = param.data.clone()
        #         optimizer.step() # not supposed to be there
                
        #         # check if the weights have changed
        #         if not torch.equal(old.data, param.data):
        #             print(f'Gradient: {param.grad}')
        #             print(param.grad.shape) # shape is [32005, 768]
        #             # print gradients not equal to 0
        #             print(f'Gradient not equal to 0: {param.grad[param.grad != 0]}')
        #             # print its shape
        #             print(f'Gradient not equal to 0 shape: {param.grad[param.grad != 0].shape}')
        #             print(f'Before: {old.data}')
        #             print("Weights have changed")
        #             print(f'After: {param.data}')
        #             print(name)
        #             print("---")
        #             input()
    
    print("Evaluating...")
    # evaluate after each epoch
    evaluation = 0
    # progress bar
    bar = progressbar.ProgressBar(max_value=100)
    # for batch in dataloader:
    for i, batch in enumerate(dataloader):
        if i> 100:
            break
        bar.update(i)
        input_ids1 = batch["input_ids1"]
        attention_mask1 = batch["attention_mask1"]
        input_ids2 = batch["input_ids2"]
        attention_mask2 = batch["attention_mask2"]
        input_ids3 = batch["input_ids3"]
        attention_mask3 = batch["attention_mask3"]
        labels = batch["label"]

        logits = hypothesis_classifier(input_ids1, attention_mask1, input_ids2, attention_mask2, input_ids3, attention_mask3)
        predictions = torch.argmax(logits, dim=1)
        evaluation += torch.sum(predictions == labels).item() / len(labels)
    evaluation /= len(dataloader)
    print("Evaluation:", evaluation)
    eval_scores.append(evaluation)

# save the embeddings model
hypothesis_classifier.save_embedding("models/hypothesis_classifier")

# save the model
torch.save(hypothesis_classifier.state_dict(), "models/hypothesis_classifier_full.pt")

# save the losses
with open("models/losses.txt", "w") as file:
    for l in losses:
        file.write(str(l) + "\n")

# save the evaluation scores
with open("models/eval_scores.txt", "w") as file:
    for e in eval_scores:
        file.write(str(e) + "\n")