import os
import progressbar
from openai import OpenAI


"""
source ~/.zshrc
"""

ihavemoney = True
if ihavemoney:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def chat(ref, hypA, hypB):
    response = client.chat.completions.create(
    model="gpt-4o", # gpt-3.5-turbo # gpt-4o
    messages=[
        {"role": "system", "content": "Une référence est une transcription exacte d'un audio. Deux hypothèses fausses sont proposées. Explique ta réflexion et finis ta phrase en écrivant 'A', 'B' ou 'a' ou 'b' si indécis."},
        {"role": "user", "content": "Référence : c' est à lui même\nHypothèse A : êtes à lui même\nHypothèse B : c' est euh à lui-même"},
        {"role": "assistant", "content": "Même si l'hypothèse B contient une disfluence ('euh'), elle correspond beaucoup mieux à la référence en termes de mots et de sens. La disfluence peut être tolérée si elle fait partie de l'original, tandis que l'erreur grammaticale de l'hypothèse A est plus problématique. Donc, la transcription la plus acceptable est l'hypothèse B."},
        {"role": "user", "content": "Référence : " + ref + "\nHypothèse A : " + hypA + "\nHypothèse B : " + hypB},
    ]
    )
    return response



def get_annotation(triplet):
    ref, hypA, hypB = triplet
    response = chat(ref, hypA, hypB)
    answer = response.choices[0].message.content # [0]
    # for each character of answer starting from the last to the first
    # until we see A or B or a or b

    choice = None
    for i in range(len(answer)):
        if answer[-i] in ['A', 'B', 'a', 'b']:
            if answer[-i] == "A":
                choice = "A"
            elif answer[-i] == "B":
                choice = "B"
            elif answer[-i] == "a":
                choice = 'a'
            elif answer[-i] == "b":
                choice = 'b'
            break
    
    if choice is None:
        print("Weird output: '" + str(answer) + "'")
        change = input("Choose A or B or a or b: ")
        if change == "A":
            choice = "A"
        elif change == "B":
            choice = "B"
        elif change == "a":
            choice = 'a'
        elif change == "b":
            choice = 'b'
        else:
            raise ValueError("Invalid choice: " + str(change))
    if choice == "A":
        return "1"
    elif choice == "B":
        return "0"
    elif choice == "a":
        return "11"
    elif choice == "b":
        return "10"


def annotate_choice(namefile):
    # check if file exists
    if os.path.exists("../datasets/" + namefile + "_gpt.txt"):
        # read the file
        with open("../datasets/" + namefile + "_gpt.txt", "r", encoding="utf8") as f:
            next(f)
            # reference hypA    hypB    annotation (0 or 1)
            data_annoted = []
            for line in f:
                line = line.strip().split("\t")[:3]
                data_annoted.append(line)

    else:
        data_annoted = []
        with open("../datasets/" + namefile + "_gpt.txt", "w", encoding="utf8") as f:
            f.write("reference\thypA\thypB\tannotation\n")

    # read the file
    with open("../datasets/" + namefile + ".txt", "r") as f:
        next(f)
        # reference hypA    hypB    annotation (0 or 1)
        data_unannoted = []
        for line in f:
            line = line.strip().split("\t")[:3]
            data_unannoted.append(line)

    # progressbar
    bar = progressbar.ProgressBar(maxval=len(data_unannoted))
    for i in range(len(data_annoted), len(data_unannoted)):
        bar.update(i)
        annotation = get_annotation(data_unannoted[i])

        with open("../datasets/" + namefile + "_gpt.txt", "a", encoding="utf8") as f:
            f.write("\t".join(data_unannoted[i]) + "\t" + annotation + "\n")



if __name__ == "__main__":
    namefile = "hats_annotation_split" # ../datasets/hats_annotation.txt
    annotate_choice(namefile)