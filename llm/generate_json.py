import json

def create_json(namefile):
    dataset = read_hats(namefile)

    txt = ""
    for i, data in enumerate(dataset):
        ref = data["reference"]
        hypA = data["hypA"]
        hypB = data["hypB"]

        dico = """{"custom_id": \"""" + str(i)
        dico += """\", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "Une référence est une transcription exacte d'un audio. Deux hypothèses fausses sont proposées. Explique ta réflexion et finis ta phrase en écrivant 'A', 'B' ou 'a' ou 'b' si indécis."},{"role": "user", "content": "Référence : c' est à lui même\\nHypothèse A : êtes à lui même\\nHypothèse B : c' est euh à lui-même"},{"role": "assistant", "content": "Même si l'hypothèse B contient une disfluence ('euh'), elle correspond beaucoup mieux à la référence en termes de mots et de sens. La disfluence peut être tolérée si elle fait partie de l'original, tandis que l'erreur grammaticale de l'hypothèse A est plus problématique. Donc, la transcription la plus acceptable est l'hypothèse B."},{"role": "user", "content": \"Référence : """
        dico += ref + """\\nHypothèse A : """ + hypA + """\\nHypothèse B : """ + hypB + """"}],"max_tokens": 3000}}"""
        txt += dico + "\n"
    with open("batch/" + namefile + ".jsonl", "w", encoding="utf8") as file:
        file.write(txt)


def read_hats(namefile):
    dataset = []
    with open("../datasets/" + namefile + ".txt", "r", encoding="utf8") as file:
        next(file)
        for line in file:
            dictionary = dict()
            line = line[:-1].split("\t")
            dictionary["reference"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["hypB"] = line[2]
            dataset.append(dictionary)
    return dataset



if __name__ == "__main__":
    namefile = "new_dataset_subset" # "new_dataset" #"temp_hats_train" # hats_test

    create_json(namefile)