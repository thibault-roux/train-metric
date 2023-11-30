import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import progressbar

from transformers import CamembertTokenizer, CamembertModel

# myenv2

# Define the Siamese network
class SiameseNetwork(nn.Module):
    def __init__(self, model_name):
        super(SiameseNetwork, self).__init__()
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model = CamembertModel.from_pretrained(model_name)
        transformer_output_dim = self.model.config.hidden_size

        self.model.train()

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
        
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

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
        print("Saving in ", path)
        self.model.save_pretrained(path)
        print("Saved")


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

# Custom dataset class
class HATSDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        ref = item['ref']
        hyp_a = item['hypA']
        hyp_b = item['hypB']
        annotation = item['annotation']

        return ref, hyp_a, hyp_b, annotation

# Function to train the Siamese network
def train():
    siamese_network = SiameseNetwork('dangvantuan/sentence-camembert-large')



    # # Check if CamembertModel parameters are in SiameseNetwork parameters
    # auto_model_params = set(siamese_network.model.named_parameters())
    # siamese_network_params = set(siamese_network.named_parameters())

    # auto_model_params_names = {name for name, _ in auto_model_params}
    # siamese_network_params_names = {name for name, _ in siamese_network_params}

    # # Check if all CamembertModel parameter names are in SiameseNetwork parameter names
    # auto_model_params_in_siamese_network = auto_model_params_names.issubset(siamese_network_params_names)

    # print(auto_model_params_names)
    # print(siamese_network_params_names)
    # print(f"CamembertModel parameters in SiameseNetwork parameters: {auto_model_params_in_siamese_network}")
    # exit()

    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
    optimizer = torch.optim.Adam(
        params=list(siamese_network.model.parameters()) + list(siamese_network.fc.parameters()),
        lr=0.999) # lr=0.001

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = train_test_split(read_hats(), test_size=0.2, random_state=42)

    train_loader = DataLoader(HATSDataset(train_dataset), batch_size=32, shuffle=True)
    val_loader = DataLoader(HATSDataset(val_dataset), batch_size=32, shuffle=False)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        print(epoch)
        siamese_network.train()
        # Training progressbar
        bar = progressbar.ProgressBar(maxval=len(train_loader))
        bar.start()
        for i, batch in enumerate(train_loader):
            if i > 1:
                break
            # for batch in train_loader:
            ref, hyp_a, hyp_b, annotation = batch
            optimizer.zero_grad()
            output = siamese_network(ref, hyp_a, hyp_b)
            loss = criterion(output, annotation.float().view(-1, 1))
            loss.backward()

            # optimizer.step()

            for name, param in siamese_network.named_parameters():
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
            bar.update(i)
            print("End")
            input()
            
        # Validation
        siamese_network.eval()
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                ref, hyp_a, hyp_b, annotation = batch
                output = siamese_network(ref, hyp_a, hyp_b)
                val_outputs.extend(output.cpu().numpy())
                val_labels.extend(annotation.numpy())

        val_outputs = torch.sigmoid(torch.tensor(val_outputs)).numpy()
        val_preds = (val_outputs > 0.5).astype(int)
        val_labels_binary = (np.array(val_labels) > 0.5).astype(int)
        val_accuracy = accuracy_score(val_labels_binary, val_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Accuracy: {val_accuracy}')

    # Save the trained model if needed
    # siamese_network.save_pretrained('models/siamese_network2.pth')
    siamese_network.save_embedding('./models/camembertmodel.pth')

if __name__ == '__main__':
    print("Lauching train.py")
    train()

