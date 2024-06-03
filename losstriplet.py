from torch.utils.data import DataLoader
import sentence_transformers
from sentence_transformers import SentenceTransformer, util, InputExample, losses


#Load the model(here we use minilm)
model = SentenceTransformer('all-MiniLM-L6-v2')


# #We get the embeddings by calling model.encode()
# emb1 = model.encode("This is a red cat with a hat.")
# emb2 = model.encode("Have you seen my red cat?")
# #Get the cosine similarity score between sentences
# cos_sim = util.cos_sim(emb1, emb2)
# print("Cosine-Similarity:", cos_sim)


#Define your train examples. You need more than just two examples...
#Inputs are wrapped around InputExample class which the model expects
#Using Triplet Loss
train_examples = [InputExample(texts=['My dear friend', 'My best friend', 'Got to hell']),
    InputExample(texts=['My good lover', 'My best lover' 'I do not want to see you'])]
#Create a PyTorch dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = sentence_transformers.losses.TripletLoss(model=model)
#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)