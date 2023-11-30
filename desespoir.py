from transformers import CamembertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import CamembertModel
from torch.utils.data import DataLoader



sentences1 = ["tu es super", "tu es méchant", "t'es cool", "t'es top", "t'es nul", "t'es bien"]
sentences2 = ["on inverse", "on inverse pas", "on inverse pas", "on inverse pas", "on inverse", "on inverse pas"]
labels = [0, 0, 1, 1, 1, 1]

# Tokenize and pad the sentences
# inputs1 = tokenizer(sentences1, padding=True, truncation=True, return_tensors="pt")
# inputs2 = tokenizer(sentences2, padding=True, truncation=True, return_tensors="pt")

class EmotionDataset(Dataset):
    def __init__(self, sentences1, sentences2, labels):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

    def __len__(self):
        return max(len(self.sentences1), len(self.sentences2))

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # return (sentences1[idx], sentences2[idx], label)
        return ({"sentence1": sentences1[idx], "sentence2": sentences2[idx], "label": label})
        # labels[idx] == 1
        # torch.tensor(self.labels[idx], dtype=torch.long) == tensor(1)


dataset = EmotionDataset(sentences1, sentences2, labels)



class EmotionClassifier(nn.Module):
    def __init__(self, camembert_model, tokenizer):
        super(EmotionClassifier, self).__init__()
        self.camembert = camembert_model
        self.tokenizer = tokenizer
        # self.linear = nn.Linear(in_features=self.camembert.config.hidden_size, out_features=2)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.camembert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    # def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
    def forward(self, ref, hyp):
        encoding1 = self.tokenizer(
            ref,
            padding='max_length',  # Pad to the maximum length in the batch
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )
        encoding2 = self.tokenizer(
            hyp,
            padding='max_length',  # Pad to the maximum length in the batch
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )
        inputs_ids1 = encoding1["input_ids"].flatten()
        attention_mask1 = encoding1["attention_mask"].flatten()
        inputs_ids2 = encoding2["input_ids"].flatten()
        attention_mask2 = encoding2["attention_mask"].flatten()

        print("inputs_ids1.shape", inputs_ids1.shape)
        print("inputs_ids1", inputs_ids1)
        print("attention_mask1", attention_mask1)
        exit()

        outputs1 = self.camembert(input_ids=inputs_ids1, attention_mask=attention_mask1)
        outputs2 = self.camembert(input_ids=inputs_ids2, attention_mask=attention_mask2)
        pooled_output1 = pooled_output1.view(pooled_output1.size(0), -1)
        pooled_output2 = pooled_output2.view(pooled_output2.size(0), -1)
        # logits = self.linear(pooled_output)
        concatenated = torch.cat([pooled_output1, pooled_output2], dim=1)
        logits = self.fc(concatenated)
        return logits

camembert_model = CamembertModel.from_pretrained("dangvantuan/sentence-camembert-large", num_labels=2)
tokenizer = CamembertTokenizer.from_pretrained("dangvantuan/sentence-camembert-large")
emotion_classifier = EmotionClassifier(camembert_model, tokenizer)




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
        print(batch)
        ref = [batch["sentence1"]]
        hyp = [batch["sentence2"]]
        labels = batch["label"]

        print()
        print("ref:", ref)
        print("labels:", labels)
        exit()

        optimizer.zero_grad()
        logits = emotion_classifier(ref, hyp)
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
