#Using Cosine SimilarityLoss
from torch.utils.data import DataLoader
#Define your train examples. You need more than just two examples...
#Inputs are wrapped around InputExample class which the model expects
#Using Triplet Loss
train_examples = [InputExample(texts=['My first sentence', 'My second sentence', 'Unrelated sentence']),
    InputExample(texts=['My first sentence', 'My second sentence' 'Unrelated sentence'])]
#Create a PyTorch dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = sentence_transformers.losses.TripletLoss(model=model)
#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)