from transformers import pipeline

model = pipeline("text-generation", "cmarkea/bloomz-560m-sft-chat")

prompt = """</s>Selon la référence, écris juste A ou B pour sélectionner la meilleure transcription parmi les deux hypothèses suivantes :
Référence : Mon radiateur est en panne
Hypothèse A : Mon mon radiateur est en panne
Hypothèse B : Mon radieux pend
Choix : <s>A
</s>Référénce : Le chat est sur le tapis
Hypothèse A : Le chien est sur le tapis
Hypothèse B : Le pain peint le papi
Choix : <s>A
</s>Référence : Je vais voir mamie
Hypothèse A : Je monte marre de pire
Hypothèse B : Je vais voir mami
Choix : <s>"""


result = model(prompt, max_new_tokens=1)

print(result)

while True:
    print()
    inp = input("Saisir votre texte : ")
    result = model(txt, max_new_tokens=1)
    print(result)