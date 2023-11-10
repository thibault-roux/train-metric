import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    def forward(self, ref, hyp_a, hyp_b):
        ref_embedding = self.embedding_model.encode(ref)
        hyp_a_embedding = self.embedding_model.encode(hyp_a)
        hyp_b_embedding = self.embedding_model.encode(hyp_b)

        concatenated = torch.cat([ref_embedding, hyp_a_embedding, hyp_b_embedding], dim=1)
        output = self.fc(concatenated)

        return output

def read_hats():
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("datasets/hats.txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["ref"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["nbrA"] = int(line[2])
            dictionary["hypB"] = line[3]
            dictionary["nbrB"] = int(line[4])
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
    num_epochs = 10
    for epoch in range(num_epochs):
        siamese_network.train()
        for batch in train_loader:
            ref, hyp_a, hyp_b, annotation = batch
            optimizer.zero_grad()
            output = siamese_network(ref, hyp_a, hyp_b)
            loss = criterion(output, annotation.float().view(-1, 1))
            loss.backward()
            optimizer.step()

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
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Accuracy: {val_accuracy}')

    # Save the trained model if needed
    torch.save(siamese_network.state_dict(), 'siamese_network.pth')

if __name__ == '__main__':
    train()
