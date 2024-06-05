import os
from openai import OpenAI
import pickle
import progressbar


# source ~/.zshrc

def chat(ref, hypA, hypB, i):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Tu écris seulement 'A' ou 'B' pour choisir la meilleure transcription."},
        {"role": "user", "content": "Référence : Mon radiateur est en panne\nHypothèse A : Mon mon radiateur est en panne\nHypothèse B : Mon radieux pend"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Référence : " + ref + "\nHypothèse A : " + hypA + "\nHypothèse B : " + hypB},
    ]
    )
    pickle.dump(response, open("pickle/" + str(i) + ".pkl", "wb"))


def read_hats(namefile):
    dataset = []
    with open("../datasets/" + namefile + ".txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            dictionary = dict()
            line = line[:-1].split("\t")
            dictionary["reference"] = line[0]
            dictionary["hypA"] = line[1]
            nbrA = int(line[2])
            dictionary["hypB"] = line[3]
            nbrB = int(line[4])
            if nbrA > nbrB:
                dictionary["best"] = "A"
            elif nbrA < nbrB:
                dictionary["best"] = "B"
            else:
                continue
            dataset.append(dictionary)
    return dataset


ihavemoney = False
if ihavemoney:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def infer(namefile):
    txt = "reference\thypA\tnbrA\thypB\tnbrB\n"
    for i, data in enumerate(dataset):
        chat(data["reference"], data["hypA"], data["hypB"], i)
        print(i)
        answer = response.choices[0].message.content
        if answer == "A":
            txt += data["reference"] + "\t" + data["hypA"] + "\t7\t" + data["hypB"] + "\t0\n"
        elif answer != "B":
            txt += data["reference"] + "\t" + data["hypA"] + "\t0\t" + data["hypB"] + "\t7\n"
        else:
            print("Weird output:", answer)

def eval(namefile):
    dataset = read_hats(namefile)
    correct = 0
    incorrect = 0

        if response.choices[0].message.content == data["best"]:
            correct += 1
        else:
            incorrect += 1
            if response.choices[0].message.content != "A" or response.choices[0].message.content != "B":
                print("Weird output:", response.choices[0].message.content)
        


if __name__ == "__main__":
    # response = pickle.load(open("response2.pkl", "rb"))
    # print(response.choices[0].message.content)

    namefile = "hats_train"