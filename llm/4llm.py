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


while True:
    print()
    ref = input("Saisir la référence: ")
    hypA = input("Saisir l'hypothèse A: ")
    hypB = input("Saisir l'hypothèse B: ")
    txt = add2prompt(prompt, ref, hypA, hypB)
    result = model(txt, max_new_tokens=1)
    print(result)