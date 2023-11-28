import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import progressbar

# myenv2

# Define the Siamese network
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model):
        super(SiameseNetwork, self).__init__()
        self.embedding_model = embedding_model
        self.fc = nn.Sequential(
            nn.Linear(3 * embedding_model.get_sentence_embedding_dimension(), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # for param in self.embedding_model.parameters():
        #     if len(param.size()) > 1:
        #         nn.init.xavier_uniform_(param.data)
        #     else:
        #         nn.init.zeros_(param.data)

    def forward(self, ref, hyp_a, hyp_b):
        ref_embedding = torch.tensor(self.embedding_model.encode(ref))
        hyp_a_embedding = torch.tensor(self.embedding_model.encode(hyp_a))
        hyp_b_embedding = torch.tensor(self.embedding_model.encode(hyp_b))

        concatenated = torch.cat([ref_embedding, hyp_a_embedding, hyp_b_embedding], dim=1)
        output = self.fc(concatenated)

        return output

    def compare_embeddings(self, embeddings_model2):
        are_models_equal = all(p1.equal(p2) for p1, p2 in zip(embeddings_model.parameters(), embeddings_model2.parameters()))
        if are_models_equal:
            print("Model weights are the same.")
        else:
            print("Model weights are different.")


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
    def __init__(self, dataset, embedding_model):
        self.dataset = dataset
        self.embedding_model = embedding_model

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
    embedding_model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    siamese_network = SiameseNetwork(embedding_model)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
    optimizer = torch.optim.Adam(siamese_network.parameters(), lr=0.001)

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = train_test_split(read_hats(), test_size=0.2, random_state=42)

    train_loader = DataLoader(HATSDataset(train_dataset, embedding_model), batch_size=32, shuffle=True)
    val_loader = DataLoader(HATSDataset(val_dataset, embedding_model), batch_size=32, shuffle=False)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        print(epoch)
        siamese_network.train()
        # Training progressbar
        bar = progressbar.ProgressBar(maxval=len(train_loader))
        bar.start()
        for i, batch in enumerate(train_loader):
            # for batch in train_loader:
            ref, hyp_a, hyp_b, annotation = batch
            optimizer.zero_grad()
            output = siamese_network(ref, hyp_a, hyp_b)
            loss = criterion(output, annotation.float().view(-1, 1))
            loss.backward()

            # Get the model parameters before the optimization step
            old_params = {name: param.clone() for name, param in siamese_network.named_parameters()}

            optimizer.step()

            # Get the model parameters after the optimization step
            new_params = {name: param for name, param in siamese_network.named_parameters()}
            # Check if any parameter has been updated
            parameters_updated = any((old_params[name] != new_params[name]).any() for name in old_params)
            if parameters_updated:
                print("Weights have been updated.")
            else:
                print("Weights have not been updated.")

            print("Check if they are different")
            siamese_network.compare_embeddings(embedding_model)
            input()

            bar.update(i)
            
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
    torch.save(embedding_model.state_dict(), 'models/fine_tuned_sentence_transformer_v2.pth')
    torch.save(siamese_network.state_dict(), 'models/siamese_network_v2.pth')

def load_model():
    embedding_model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    siamese_network = SiameseNetwork(embedding_model)
    siamese_network.load_state_dict(torch.load('models/siamese_network.pth'))
    return siamese_network

if __name__ == '__main__':
    print("Lauching train.py")
    train()
