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


def get_dataloader(namefile):
    # train_examples = [InputExample(texts=['My dear friend', 'My best friend', 'Got to hell']),
    #     InputExample(texts=['My first sentence', 'My second sentence' 'Unrelated sentence'])]

    train_examples = []
    with open("datasets/" + namefile + ".txt", "r", encoding="utf8") as file:
        next(file) # reference  hypA    nbrA    hypB    nbrB
        for line in file:
            line = line[:-1].split("\t")
            reference = line[0]
            hypA = line[1]
            hypB = line[3]
            nbrA = int(line[2])
            nbrB = int(line[4])
            if nbrA > nbrB:
                train_examples.append(InputExample(texts=[reference, hypA, hypB]))
            elif nbrA < nbrB:
                train_examples.append(InputExample(texts=[reference, hypB, hypA]))

    #Create a PyTorch dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)



train_loss = sentence_transformers.losses.TripletLoss(model=model)
#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)