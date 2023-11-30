from transformers import CamembertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import CamembertModel
from torch.utils.data import DataLoader


tokenizer = CamembertTokenizer.from_pretrained("camembert-base")


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
            dictionary["annotation"] = float(line[3])
            dataset.append(dictionary)
    return dataset


class HATSDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.sentences1 = []
        self.sentences2 = []
        self.labels = []
        for item in dataset:
            self.sentences1.append(item['ref'])
            self.sentences2.append(item['hypA'])
            # self.sentences3.append(item['hypB'])
            self.labels.append(item['annotation'])

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return max(len(self.sentences1), len(self.sentences2))

    def __getitem__(self, idx):
        encoding1 = self.tokenizer(
            self.sentences1[idx],
            padding='max_length',  # Pad to the maximum length in the batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding2 = self.tokenizer(
            self.sentences2[idx],
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
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        })

# set dataset
max_length = 10
hats = read_hats()
dataset = HATSDataset(hats, tokenizer, max_length)



class HypothesisClassifier(nn.Module):
    def __init__(self, camembert_model):
        super(HypothesisClassifier, self).__init__()
        self.camembert = camembert_model
        # self.linear = nn.Linear(in_features=self.camembert.config.hidden_size, out_features=2)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.camembert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.camembert(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.camembert(input_ids=input_ids2, attention_mask=attention_mask2)
        pooled_output1 = outputs1.pooler_output
        pooled_output2 = outputs1.pooler_output
        # logits = self.linear(pooled_output)
        concatenated = torch.cat([pooled_output1, pooled_output2], dim=1)
        logits = self.fc(concatenated)
        return logits

camembert_model = CamembertModel.from_pretrained("camembert-base", num_labels=2)
emotion_classifier = HypothesisClassifier(camembert_model)




# Define your training parameters
optimizer = torch.optim.Adam(emotion_classifier.parameters(), lr=1)
loss_fn = nn.CrossEntropyLoss()

# DataLoader for training
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    print(epoch)
    for batch in dataloader:
        input_ids1 = batch["input_ids1"]
        attention_mask1 = batch["attention_mask1"]
        input_ids2 = batch["input_ids2"]
        attention_mask2 = batch["attention_mask2"]
        labels = batch["label"]

        optimizer.zero_grad()
        logits = emotion_classifier(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = loss_fn(logits, labels)
        loss.backward()
        # optimizer.step()

        for name, param in emotion_classifier.named_parameters():
            if param.requires_grad:
                old = param.data.clone()
                optimizer.step() # not supposed to be there
                
                # check if the weights have changed
                if not torch.equal(old.data, param.data):
                    print(f'Gradient: {param.grad}')
                    print(f'Before: {old.data}')
                    print("Weights have changed")
                    print(f'After: {param.data}')
                    print(name)
                    print("---")
                    input()

# Save the fine-tuned model
# emotion_classifier.save_pretrained("emotion_model")