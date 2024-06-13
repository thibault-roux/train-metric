import os
from openai import OpenAI
import pickle
import progressbar

"""
source ~/.zshrc
"""

def chat(ref, hypA, hypB, i):
    # check if the pickle exists
    if os.path.exists("pickle/pickle8/" + str(i) + ".pkl"):
        return pickle.load(open("pickle/pickle8/" + str(i) + ".pkl", "rb"))
    else:
        response = client.chat.completions.create(
        model="gpt-4o", # gpt-3.5-turbo # gpt-4o
        messages=[
            {"role": "system", "content": "Nous appelons référence une transcription exacte d'un audio vers du texte. Deux hypothèses produites par des systèmes de reconnaissances de la parole. Tu expliques d'abord ton raisonnement et tu finis ta réponse par 'A' ou 'B' pour choisir la transcription qui te paraît la plus acceptable d'un point de vue sémantique ou phonétique."},
            {"role": "user", "content": "Référence : c' est à lui même\nHypothèse A : êtes à lui même\nHypothèse B : c' est euh à lui-même"},
            {"role": "assistant", "content": "Même si l'hypothèse B contient une disfluence ('euh'), elle correspond beaucoup mieux à la référence en termes de mots et de sens. La disfluence peut être tolérée si elle fait partie de l'original, tandis que l'erreur grammaticale de l'hypothèse A est plus problématique. Donc, la transcription la plus acceptable est l'hypothèse B."},
            {"role": "user", "content": "Référence : " + ref + "\nHypothèse A : " + hypA + "\nHypothèse B : " + hypB},
        ]
        )
        pickle.dump(response, open("pickle/pickle8/" + str(i) + ".pkl", "wb"))
        return response


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

def infer(namefile, maxval=3614):
    dataset = read_hats(namefile)
    txt = "reference\thypA\tnbrA\thypB\tnbrB\n"
    # progressbar
    bar = progressbar.ProgressBar(maxval=len(dataset))
    bar.start()
    for i, data in enumerate(dataset):
        bar.update(i)
        response = chat(data["reference"], data["hypA"], data["hypB"], i)
        print(i)
        answer = response.choices[0].message.content # [0]
        if "A" in answer[-3:]:
            answer = "A"
        elif "B" in answer[-3:]:
            answer = "B"
        # else:
        #     print("Weird output: '" + str(answer) + "'")
        #     change = input("Choose A or B: ")
        #     if change == "A":
        #         answer = "A"
        #     elif change == "B":
        #         answer = "B"
        if answer == "A":
            txt += data["reference"] + "\t" + data["hypA"] + "\t7\t" + data["hypB"] + "\t0\n"
        elif answer == "B":
            txt += data["reference"] + "\t" + data["hypA"] + "\t0\t" + data["hypB"] + "\t7\n"
        if i == maxval:
            break
    with open("../datasets/" + namefile + "_chatgpt.txt", "w", encoding="utf8") as file:
        file.write(txt)
    print("File writed.")
    bar.finish()

def eval(namefile, maxval):
    print("evaluating...")
    correct_dataset = read_hats(namefile)
    infered_dataset = read_hats(namefile + "_chatgpt")

    correct = 0
    incorrect = 0
    for i, data in enumerate(correct_dataset):
        if data["best"] == infered_dataset[i]["best"]:
            correct += 1
        else:
            incorrect += 1
        if i == maxval:
            break
    print("Correct:", correct)
    print("Incorrect:", incorrect)
    print("Accuracy:", correct / (correct + incorrect))



if __name__ == "__main__":
    # response = pickle.load(open("response2.pkl", "rb"))
    # print(response.choices[0].message.content)

    namefile = "hats_extended"
    infer(namefile, maxval=3614)
    eval(namefile, maxval=3614)