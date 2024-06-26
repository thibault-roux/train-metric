from torch.utils.data import DataLoader
import sentence_transformers
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity





def get_dataloader(namefile, batch_size=64):
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
            nbrA = float(line[2])
            nbrB = float(line[4])
            if nbrA > nbrB:
                train_examples.append(InputExample(texts=[reference, hypA, hypB]))
            elif nbrA < nbrB:
                train_examples.append(InputExample(texts=[reference, hypB, hypA]))

    #Create a PyTorch dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    return train_dataloader


# ----------------- Evaluator -----------------
def semdist(ref, hyp, model):
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score)*100 # lower is better

def evaluate(model, namefile, filter=0):
    print("Evaluating...")
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
            agreement = max(nbrA, nbrB) / (nbrA + nbrB)
            if agreement <= filter:
                continue
            scoreA = semdist(reference, hypA, model)
            scoreB = semdist(reference, hypB, model)
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
    # model = SentenceTransformer('dangvantuan/sentence-camembert-base')
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    train_dataloader = get_dataloader("hats_train_plus_gpt") # hats_train, hats_train_best, hats_extended, hats_extended_chatgpt, hats_train_plus_gpt
    train_loss = sentence_transformers.losses.TripletLoss(model=model)

    # evaluate(model, "hats_test") # hats_test, hats_test_best # 0.7162698412698413
    
    # Tune the model
    # print("Fine tuning...")
    # model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=10) #, optimizer_params={'lr': 2e-6}) # train_best, epochs=3, warmup_steps=10 worked well: 72.61
    # save in models/losstriplet
    # model.save("models/losstriplet2")

    # Test the model
    evaluate(model, "hats_test", 1) # hats_test, hats_test_best