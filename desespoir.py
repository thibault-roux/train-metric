from transformers import CamembertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import CamembertModel
from torch.utils.data import DataLoader


tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

sentences1 = ["tu es super", "tu es méchant", "t'es cool", "t'es top", "t'es nul", "t'es bien"]
sentences2 = ["on inverse", "on inverse pas", "on inverse pas", "on inverse pas", "on inverse", "on inverse pas"]
labels = [0, 0, 1, 1, 1, 1]

# Tokenize and pad the sentences
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

class EmotionDataset(Dataset):
    def __init__(self, sentences1, sentences2, labels, tokenizer, max_length):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

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
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        },
        {
            "input_ids": encoding2["input_ids"].flatten(),
            "attention_mask": encoding2["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        })

# Set your desired maximum sequence length
max_length = 10

dataset = EmotionDataset(sentences, labels, tokenizer, max_length)



class EmotionClassifier(nn.Module):
    def __init__(self, camembert_model):
        super(EmotionClassifier, self).__init__()
        self.camembert = camembert_model
        # self.linear = nn.Linear(in_features=self.camembert.config.hidden_size, out_features=2)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.camembert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.camembert(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.camembert(input_ids=input_ids2, attention_mask=attention_mask2)
        pooled_output1 = outputs1.pooler_output
        pooled_output1 = outputs1.pooler_output
        # logits = self.linear(pooled_output)
        concatenated = torch.cat([pooled_output1, pooled_output2], dim=1)
        logits = self.fc(concatenated)
        return logits

camembert_model = CamembertModel.from_pretrained("camembert-base", num_labels=2)
emotion_classifier = EmotionClassifier(camembert_model)




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
        batch1, batch2 = batch
        input_ids1 = batch1["input_ids"]
        attention_mask1 = batch1["attention_mask"]
        # labels1 = batch1["label"]
        input_ids2 = batch2["input_ids"]
        attention_mask2 = batch2["attention_mask"]
        labels = batch1["label"]

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
