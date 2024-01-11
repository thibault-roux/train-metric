from transformers import CamembertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import CamembertModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import progressbar

tokenizer = CamembertTokenizer.from_pretrained('dangvantuan/sentence-camembert-base')

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






class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_model_name, max_length):
        super(SiameseNetwork, self).__init__()

        # CamemBERT model
        self.camembert = CamembertModel.from_pretrained(pretrained_model_name)

        # Projection layer for sentence embeddings
        self.projection_layer = nn.Sequential(
            nn.Linear(self.camembert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.camembert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation

        # Project embeddings through the projection layer
        projected_embeddings = self.projection_layer(embeddings)

        return projected_embeddings

class SiameseNetworkWithMarginLoss(nn.Module):
    def __init__(self, siamese_network):
        super(SiameseNetworkWithMarginLoss, self).__init__()
        self.siamese_network = siamese_network
        self.margin_loss = nn.MarginRankingLoss(margin=0.5)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, input_ids3, attention_mask3, label):
        output1 = self.siamese_network(input_ids1, attention_mask1)
        output2 = self.siamese_network(input_ids2, attention_mask2)
        output3 = self.siamese_network(input_ids3, attention_mask3)

        loss = self.margin_loss(output1, output2, label) + self.margin_loss(output1, output3, label)

        return loss


def evaluate_siamese_network(siamese_network, dataloader):
    siamese_network.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
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

            similarity_pos = nn.functional.cosine_similarity(output1, output2)
            similarity_neg = nn.functional.cosine_similarity(output1, output3)
            predictions = (similarity_pos > similarity_neg).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

# Set up the Siamese network and the dataset
pretrained_model_name = 'dangvantuan/sentence-camembert-base'
max_length = 30
siamese_network = SiameseNetwork(pretrained_model_name, max_length)
siamese_with_margin_loss = SiameseNetworkWithMarginLoss(siamese_network)
hats_dataset = HATSDataset(hats, tokenizer, max_length)


# Set up data loader
batch_size = 16
dataloader = DataLoader(hats_dataset, batch_size=batch_size, shuffle=True)

# Set up data loader for evaluation
eval_dataloader = DataLoader(hats_dataset, batch_size=batch_size, shuffle=False)

# Set up optimizer
optimizer = Adam(siamese_with_margin_loss.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
losses = []
for epoch in range(num_epochs):
    print(epoch)
    bar = progressbar.ProgressBar(max_value=len(dataloader))
    for i, batch in enumerate(dataloader):
        bar.update(i)
        if i > 100:
            break
        optimizer.zero_grad()
        loss = siamese_with_margin_loss(**batch)
        loss.backward()
        losses.append(loss.item())
        print(loss.item())
        optimizer.step()

    # Evaluation
    accuracy = evaluate_siamese_network(siamese_network, eval_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}")

print("losses:", losses)

# Save the fine-tuned model
siamese_network.save_pretrained('fine_tuned_camembert_hats_model')
