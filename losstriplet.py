from torch.utils.data import DataLoader
import sentence_transformers
from sentence_transformers import SentenceTransformer, util, InputExample, losses





def get_dataloader(namefile, batch_size=16):
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
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    return train_dataloader


# ----------------- Evaluator -----------------
def evaluate(model, namefile):
    correct = 0
    incorrect = 0
    with open("datasets/" + namefile + ".txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            reference = line[0]
            hypA = line[1]
            hypB = line[3]
            nbrA = int(line[2])
            nbrB = int(line[4])
            scoreA = model.similarity(reference, hypA)
            scoreB = model.similarity(reference, hypB)
            if scoreA < scoreB and nbrA > nbrB:
                correct += 1
            elif scoreA > scoreB and nbrA < nbrB:
                correct += 1
            else:
                incorrect += 1

    print("Correct:", correct)
    print("Incorrect:", incorrect)
    print("Accuracy:", correct/(correct+incorrect))



if __name__ == "__main__":
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    train_dataloader = get_dataloader("hats_train") # hats_train, hats_train_best
    train_loss = sentence_transformers.losses.TripletLoss(model=model)

    
    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    # save in models/losstriplet
    model.save("models/losstriplet")

    # Test the model
    model = SentenceTransformer("models/losstriplet")
    evaluate(model, "hats_test") # hats_test, hats_test_best