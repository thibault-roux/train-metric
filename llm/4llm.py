from transformers import pipeline

model = pipeline("text-generation", "cmarkea/bloomz-560m-sft-chat")

prompt = """</s>Selon la référence, écris juste A ou B pour sélectionner la meilleure transcription parmi les deux hypothèses suivantes :
Référence : Mon radiateur est en panne
Hypothèse A : Mon mon radiateur est en panne
Hypothèse B : Mon radieux pend
Choix : <s>A
</s>Référénce : Le chat est sur le tapis
Hypothèse A : Le pain peint le papi
Hypothèse B : Le chien est sur le tapis
Choix : <s>B
</s>Référence : Je vais voir mamie
Hypothèse A : Je monte marre de pire
Hypothèse B : Je vais voir mami
Choix : <s>B"""

def add2prompt(prompt, ref, hypA, hypB):
    prompt += "\n</s>Référence : " + ref + "\nHypothèse A : " + hypA + "\nHypothèse B : " + hypB  + "\nChoix : <s>"
    return prompt


def chat(prompt):
    ref = input("Saisir la référence: ")
    hypA = input("Saisir l'hypothèse A: ")
    hypB = input("Saisir l'hypothèse B: ")
    txt = add2prompt(prompt, ref, hypA, hypB)
    result = model(txt, max_new_tokens=1)
    print(result)
    return result[0]['generated_text'][:-1]

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


if __name__ == "__main__":
    dataset = read_hats("hats_train")
    for data in dataset:
        prompt = add2prompt(prompt, data["reference"], data["hypA"], data["hypB"])
        response = chat(prompt)
        if response == data["best"]:
            print("Correct")
        else:
            print("Incorrect")
        input()