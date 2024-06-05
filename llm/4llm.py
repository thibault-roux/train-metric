from transformers import pipeline

model = pipeline("text-generation", "cmarkea/bloomz-560m-sft-chat")

# txt = "</s>C'est quoi le deep learning ?<s>"
txt = "</s>Selon la référence, écris juste A ou B pour sélectionner la meilleure transcription parmi les deux hypothèses suivantes :\nRéférence : Mon radiateur est en panne\tHypothèse A : Mon mon radiateur est en panne\tHypothèse B : Mon radieux pend\t</s>"
result = model(txt, max_new_tokens=512)

print(result)

while True:
    print()
    txt = input("Saisir votre texte : ")
    result = model(txt, max_new_tokens=512)
    print(result)